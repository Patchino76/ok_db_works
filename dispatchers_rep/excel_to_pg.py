import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import re
import sys

def extract_year_from_filename(filename):
    """
    Extract the year from a filename that contains a year in the format 20XX.
    If no year is found, returns the current year.
    
    Args:
        filename (str): The filename to extract the year from
        
    Returns:
        int: The extracted year
    """
    # Extract the filename from the path if it contains directory separators
    base_filename = os.path.basename(filename)
    
    # Look for a 4-digit year pattern (20XX) in the filename
    year_pattern = r'20\d{2}'  # Matches 2000-2099
    match = re.search(year_pattern, base_filename)
    
    if match:
        return int(match.group(0))
    else:
        # If no year found in the filename, return current year
        return datetime.now().year

class ExcelToPGConverter:
    """
    Class to read Excel dispatcher data and insert it directly into PostgreSQL.
    
    This combines functionality from:
    - excel_to_csv_converter.py: For Excel file processing logic
    - scheduled_pulse_postgresql.py: For PostgreSQL connection parameters
    """
    
    def __init__(self, pg_host, pg_port, pg_dbname, pg_user, pg_password):
        """Initialize with PostgreSQL connection parameters"""
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_dbname = pg_dbname
        self.pg_user = pg_user
        self.pg_password = pg_password
        
    def connect_to_postgresql(self):
        """Create and return a new connection to PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                dbname=self.pg_dbname,
                user=self.pg_user,
                password=self.pg_password
            )
            return conn
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None
    
    def create_table_from_dataframe(self, df):
        """
        Drop existing table and create a new one based on DataFrame columns.
        This ensures the table structure perfectly matches our data.
        
        Args:
            df (pandas.DataFrame): DataFrame with processed data
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.connect_to_postgresql()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Start a transaction
            conn.autocommit = False
            
            # Drop the table if it exists (with schema)
            print("Dropping existing ore_quality table if exists...")
            cursor.execute("DROP TABLE IF EXISTS mills.ore_quality CASCADE;")
            conn.commit()
            print("Table dropped successfully")
            
            # Create new table with columns derived from the DataFrame
            print("Creating new ore_quality table based on DataFrame structure...")
            
            # Build the create table SQL dynamically based on DataFrame columns
            columns = []
            columns.append("id SERIAL PRIMARY KEY")
            columns.append("date DATE NOT NULL")
            columns.append("shift INTEGER NOT NULL")
            columns.append("timestamp TIMESTAMP NOT NULL")  # Always include timestamp
            
            # Map DataFrame columns to PostgreSQL data types
            df_columns = list(df.columns)
            for col in df_columns:
                # Skip date and shift as they're already added
                if col in ['date', 'shift', 'timestamp']:
                    continue
                    
                # Add other columns as NUMERIC (for ore quality data) or VARCHAR (for text)
                if col in ['class_15', 'class_12', 'grano', 'daiki', 'shisti']:
                    columns.append(f"{col} NUMERIC")
                else:
                    columns.append(f"{col} VARCHAR(100)")  # Default for other columns
            
            # Always add created_at column
            columns.append("created_at TIMESTAMP DEFAULT NOW()")
            
            # Create the table in the mills schema
            create_table_query = f"""
            CREATE TABLE mills.ore_quality (
                {', '.join(columns)}
            );
            """
            
            print("SQL for table creation:")
            print(create_table_query)
            
            try:
                cursor.execute(create_table_query)
                conn.commit()
                print("Table 'ore_quality' created successfully with columns:")
                for col in columns:
                    print(f"  - {col}")
                return True
            except Exception as e:
                print(f"Error creating table: {e}")
                conn.rollback()
                return False
        except Exception as e:
            print(f"Error recreating table: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()
                
    def process_excel_to_pg(self, input_excel):
        """
        Process Excel file and insert data into PostgreSQL.
        
        Args:
            input_excel (str): Path to the input Excel file
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"Processing Excel file: {input_excel}")
        
        # Extract year from the filename or use current year as fallback
        file_year = extract_year_from_filename(input_excel)
        print(f"Extracted year from filename: {file_year}")
        
        # Create an empty list to hold all data
        all_data = []
        
        # Read all sheets
        try:
            excel_file = pd.ExcelFile(input_excel)
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return False
        
        # Define the columns we want in our final output - using English names
        required_columns = ['date', 'shift', 'class_15', 'class_12', 'grano', 'daiki', 'shisti']
        
        # Mapping between Bulgarian column names in Excel and English names for our processing
        column_name_mapping = {
            'класа +15мм': 'class_15',
            'класа +12.5мм': 'class_12', 
            'гранодиорити': 'grano',
            'дайки': 'daiki',
            'шисти': 'shisti'
        }
        
        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            print(f"Processing sheet: {sheet_name}")
            
            # Read the sheet data
            df = pd.read_excel(input_excel, sheet_name=sheet_name)
            
            # Use the year extracted from the filename
            current_year = file_year
            
            # Get month number from sheet name (Bulgarian month names)
            month_mapping = {
                'Януари': 1, 'Февруари': 2, 'Март': 3, 'Април': 4, 
                'Май': 5, 'Юни': 6, 'Юли': 7, 'Август': 8,
                'Септември': 9, 'Октомври': 10, 'Ноември': 11, 'Декември': 12
            }
            month = month_mapping.get(sheet_name, datetime.now().month)
            print(f"Sheet '{sheet_name}' corresponds to month {month}")
            
            # Find column indices
            date_col = 'Дата'
            shift_col = 'Смяна'
            
            # Verify critical columns exist
            if date_col not in df.columns or shift_col not in df.columns:
                print(f"Warning: Required columns not found in sheet {sheet_name}. Skipping.")
                continue
            
            # Find the target columns - using flexible matching to handle different formats between sheets
            target_columns = {}
            for col in df.columns:
                col_str = str(col).lower()
                
                # class_15 (Класа +15мм) - look for +15 in any format
                if '+15' in col_str:
                    target_columns['class_15'] = col
                    
                # class_12 (Класа +12.5мм) - key column that was missing, be more flexible with matching
                elif any(pattern in col_str for pattern in ['+12.5', '+12,5', '+12']):
                    target_columns['class_12'] = col
                    
                # grano (Гранодиорити)
                elif 'грано' in col_str:
                    target_columns['grano'] = col
                    
                # daiki (Дайки)
                elif 'дайки' in col_str:
                    target_columns['daiki'] = col
                    
                # shisti (Шисти)
                elif 'шисти' in col_str:
                    target_columns['shisti'] = col
            
            print(f"Target columns found: {target_columns}")
            
            # Check if all required target columns were found
            missing_cols = [col for col in ['class_15', 'class_12', 'grano', 'daiki', 'shisti'] 
                           if col not in target_columns]
            if missing_cols:
                print(f"Warning: Missing columns in sheet {sheet_name}: {missing_cols}")
                
            # Process data row by row
            current_date = None
            
            # Iterate through rows
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Update current date if available in this row
                if not pd.isna(row[date_col]) and isinstance(row[date_col], (int, float)):
                    current_date = int(row[date_col])
                    
                # Skip rows with no shift info or with 'Общо' in the shift column
                if pd.isna(row[shift_col]) or not isinstance(row[shift_col], (int, float)) or int(row[shift_col]) not in [1, 2, 3]:
                    continue
                    
                # Skip if we don't have a valid date
                if current_date is None:
                    continue
                    
                # Get the shift number
                shift = int(row[shift_col])
                    
                # Create a proper date - with validation
                try:
                    # Check if the current_date is valid for this month
                    # Get the number of days in the month
                    import calendar
                    max_days = calendar.monthrange(current_year, month)[1]
                    
                    if current_date <= 0 or current_date > max_days:
                        print(f"Warning: Invalid day {current_date} for month {month}. Skipping...")
                        continue
                        
                    # Create a valid date string for PostgreSQL
                    date_str = f"{current_year:04d}-{month:02d}-{current_date:02d}"
                    
                    # Generate proper timestamps based on shifts
                    # Shift 1: 06:00, Shift 2: 14:00, Shift 3: 22:00
                    shift_hours = {1: 6, 2: 14, 3: 22}
                    shift_hour = shift_hours.get(shift, 0)
                    timestamp = f"{date_str} {shift_hour:02d}:00:00"
                    
                except Exception as e:
                    print(f"Warning: Invalid date - Year: {current_year}, Month: {month}, Day: {current_date}. Error: {e}")
                    continue
                    
                # Create row data with all required fields
                row_data = {
                    'date': date_str,
                    'shift': shift,
                    'timestamp': timestamp,
                    'original_sheet': sheet_name
                }
                    
                # Add target column values
                for col_name, original_col in target_columns.items():
                    try:
                        value = row[original_col]
                        if pd.isna(value):
                            row_data[col_name] = None  # Use None instead of np.nan for PostgreSQL
                        elif isinstance(value, (int, float)):
                            row_data[col_name] = value
                        else:
                            # Try to convert to float if it's a string
                            try:
                                row_data[col_name] = float(str(value).replace(',', '.'))
                            except:
                                row_data[col_name] = None
                    except:
                        row_data[col_name] = None
                
                # Make sure all required columns have a value (even if None)
                for col in required_columns:
                    if col not in row_data:
                        row_data[col] = None
                        
                # Add to our data collection
                all_data.append(row_data)
        
        # Check if we have data
        if not all_data:
            print("No valid data found to process!")
            return False
            
        # Create dataframe from all collected data
        result_df = pd.DataFrame(all_data)
        
        # Ensure all required columns are present
        for col in required_columns:
            if col not in result_df.columns:
                result_df[col] = None
        
        # Sort by date and shift
        result_df.sort_values(['date', 'shift'], inplace=True)
        
        # Ensure columns are in the correct order for PostgreSQL insert
        required_column_order = ['date', 'shift', 'timestamp', 'class_15', 'class_12', 'grano', 'daiki', 'shisti', 'original_sheet']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_column_order if col not in result_df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in DataFrame: {missing_columns}")
            for col in missing_columns:
                result_df[col] = None  # Add missing columns with None values
                
        # Reorder columns to match the expected order in PostgreSQL
        print("Original DataFrame columns:", result_df.columns.tolist())
        result_df = result_df[required_column_order]
        print("Reordered DataFrame columns:", result_df.columns.tolist())
        
        # First create/recreate the table based on the DataFrame structure
        if not self.create_table_from_dataframe(result_df):
            print("Failed to create or update table structure")
            return False
            
        # Now insert the data
        return self.insert_into_postgresql(result_df)
        
    def insert_into_postgresql(self, df):
        """
        Insert the processed DataFrame into PostgreSQL table.
        
        Args:
            df (pandas.DataFrame): DataFrame with processed data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate DataFrame
        if df.empty:
            print("Error: DataFrame is empty, no data to insert")
            return False

        # Check that all required columns exist
        required_columns = ['date', 'shift', 'timestamp', 'class_15', 'class_12', 'grano', 'daiki', 'shisti']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns in DataFrame: {missing_columns}")
            return False

        # Connect to PostgreSQL
        conn = self.connect_to_postgresql()
        if not conn:
            return False

        try:
            # Prepare data for insert
            data_tuples = []

            # Debug: Show DataFrame columns
            print("DataFrame columns:", df.columns.tolist())
            print(f"Number of rows to process: {len(df)}")

            for idx, row in df.iterrows():
                try:
                    # Convert pandas date string to Python date object
                    date_obj = datetime.strptime(row['date'], '%Y-%m-%d').date()

                    # Handle NaN/None values - convert them to None for PostgreSQL
                    class_15 = None if pd.isna(row['class_15']) else row['class_15']
                    class_12 = None if pd.isna(row['class_12']) else row['class_12']
                    grano = None if pd.isna(row['grano']) else row['grano']
                    daiki = None if pd.isna(row['daiki']) else row['daiki']
                    shisti = None if pd.isna(row['shisti']) else row['shisti']
                    
                    # Create tuple of values in the order they appear in the table
                    data_tuple = (
                        date_obj,                           # date
                        int(row['shift']),                  # shift
                        datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),  # timestamp
                        class_15,                          # class_15
                        class_12,                          # class_12
                        grano,                             # grano
                        daiki,                             # daiki
                        shisti,                            # shisti
                        row['original_sheet']               # original_sheet
                    )
                    data_tuples.append(data_tuple)
                    
                except Exception as row_error:
                    print(f"Warning: Error processing row {idx}: {row_error}")
                    print(f"Row data: {row.to_dict()}")
                    continue
                    
            # Check if we have data to insert
            if not data_tuples:
                print("Error: No valid data tuples to insert")
                return False
                
            # Execute batch insert into mills schema
            cursor = conn.cursor()
            insert_query = """
            INSERT INTO mills.ore_quality (date, shift, timestamp, class_15, class_12, grano, daiki, shisti, original_sheet)
            VALUES %s
            """
            
            print(f"Preparing to insert {len(data_tuples)} rows")
            
            # Use psycopg2.extras.execute_values for efficient batch insert
            execute_values(cursor, insert_query, data_tuples)
            
            # Commit changes
            conn.commit()
            
            print(f"Successfully inserted {len(data_tuples)} rows into ore_quality table")
            return True
            
        except Exception as e:
            print(f"Error inserting data into PostgreSQL: {e}")
            print("This is often caused by a mismatch between the DataFrame columns and table columns")
            if conn:
                conn.rollback()
            return False
            
        finally:
            if conn:
                conn.close()
