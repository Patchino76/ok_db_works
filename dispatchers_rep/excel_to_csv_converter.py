import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_excel_to_csv(input_excel, output_csv):
    """
    Process Excel file with dispatcher data and convert it to CSV.
    
    The Excel file contains multiple sheets (one per month) with shift data.
    This function:
    1. Extracts data from all sheets
    2. Processes each day's shifts (1, 2, 3)
    3. Skips rows with totals ("Общо")
    4. Combines all data into a single CSV
    
    Args:
        input_excel (str): Path to the input Excel file
        output_csv (str): Path where the output CSV will be saved
    
    Returns:
        pandas.DataFrame: Processed data
    """
    print(f"Processing Excel file: {input_excel}")
    
    # Create an empty list to hold all data
    all_data = []
    
    # Read all sheets
    excel_file = pd.ExcelFile(input_excel)
    
    # Define the columns we want in our final output
    required_columns = ['date', 'shift', 'класа +15мм', 'класа +12.5мм', 'гранодиорити', 'дайки', 'шисти']
    
    # Process each sheet
    for sheet_name in excel_file.sheet_names:
        print(f"Processing sheet: {sheet_name}")
        
        # Read the sheet data
        df = pd.read_excel(input_excel, sheet_name=sheet_name)
        
        # Get the year from current date (could be parameterized)
        current_year = datetime.now().year
        
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
        
        # Print all columns for debugging
        print("Available columns in this sheet:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col!r}")
        
        # Match target columns using flexible pattern matching
        for col in df.columns:
            col_str = str(col).lower()
            
            # Класа +15мм - look for +15 in any format
            if '+15' in col_str:
                target_columns['класа +15мм'] = col
                
            # Класа +12.5мм - key column that was missing, be more flexible with matching
            elif any(pattern in col_str for pattern in ['+12.5', '+12,5', '+12']):
                target_columns['класа +12.5мм'] = col
                
            # Гранодиорити
            elif 'грано' in col_str:
                target_columns['гранодиорити'] = col
                
            # Дайки
            elif 'дайки' in col_str:
                target_columns['дайки'] = col
                
            # Шисти
            elif 'шисти' in col_str:
                target_columns['шисти'] = col
        
        print(f"Target columns found: {target_columns}")
        
        # Check if all required target columns were found
        missing_cols = [col for col in ['класа +15мм', 'класа +12.5мм', 'гранодиорити', 'дайки', 'шисти'] 
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
                
            # Create a proper date
            try:
                date_str = f"{current_year:04d}-{month:02d}-{current_date:02d}"
            except Exception as e:
                print(f"Warning: Invalid date - Year: {current_year}, Month: {month}, Day: {current_date}. Error: {e}")
                continue
                
            # Create row data with all required fields
            row_data = {
                'date': date_str,
                'shift': shift,
                'sheet': sheet_name  # Keep track of source sheet
            }
                
            # Add target column values
            for col_name, original_col in target_columns.items():
                try:
                    value = row[original_col]
                    if pd.isna(value):
                        row_data[col_name] = np.nan
                    elif isinstance(value, (int, float)):
                        row_data[col_name] = value
                    else:
                        # Try to convert to float if it's a string
                        try:
                            row_data[col_name] = float(str(value).replace(',', '.'))
                        except:
                            row_data[col_name] = np.nan
                except:
                    row_data[col_name] = np.nan
            
            # Make sure all required columns have a value (even if NaN)
            for col in required_columns:
                if col not in row_data:
                    row_data[col] = np.nan
                    
            # Add to our data collection
            all_data.append(row_data)
    
    # Check if we have data
    if not all_data:
        print("No valid data found to process!")
        return None
        
    # Create dataframe from all collected data
    result_df = pd.DataFrame(all_data)
    
    # Ensure all required columns are present
    for col in required_columns:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    # Sort by date and shift
    result_df.sort_values(['date', 'shift'], inplace=True)
    
    # Drop the sheet column as it was just for reference
    if 'sheet' in result_df.columns:
        result_df.drop('sheet', axis=1, inplace=True)
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Report saved to {output_csv}")
    
    return result_df

def main():
    """
    Main function to run the Excel to CSV conversion.
    
    This can be called directly or imported and used in other scripts.
    """
    # Get base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Get input/output paths
    input_file = os.path.join(base_dir, 'dispatchers_rep', 'report.xlsx')
    output_file = os.path.join(base_dir, 'dispatchers_rep', 'report.csv')
    
    print(f"Using input file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    # Process the data
    result = process_excel_to_csv(input_file, output_file)
    
    # Display sample of the result
    if result is not None and not result.empty:
        print("\nSample of the generated report:")
        print(result.head(10))
        
        # Also print column information to verify all data is present
        print("\nColumn statistics:")
        for col in result.columns:
            non_null_count = result[col].count()
            total_count = len(result)
            print(f"  {col}: {non_null_count}/{total_count} non-null values ({non_null_count/total_count:.1%})")
    else:
        print("\nNo data was processed. Please check the input file and error messages above.")

if __name__ == "__main__":
    main()
