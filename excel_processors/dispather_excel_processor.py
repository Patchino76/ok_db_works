import pandas as pd
from datetime import datetime, timedelta
import os
import calendar
import re

"""
Excel Dispatcher Data Processor

This module provides functionality to convert the dispatcher Excel file with 12 monthly sheets
into a pandas DataFrame with proper timestamps.

The main function `excel_to_dataframe` loads the Excel file, processes each sheet,
and returns a combined DataFrame with:
- All original columns from the Excel file
- A "TimeStamp" column with proper datetime values (YYYY-MM-DD HH:MM:SS)
- The "Дата" column updated to reflect the correct date in DD.MM.YYYY format
- An "OriginalSheet" column indicating the source sheet

The "TimeStamp" column is created based on:
- Year: Extracted from the Excel file name (e.g., "Doklad_Dispecheri_2024.xlsx" -> 2024)
- Month: From the sheet name
- Day: Sequential days in the month
- Time: Based on shift (1: 06:00, 2: 14:00, 3: 22:00)

The module also filters out rows containing "Общо" in the "Смяна" column.
"""

def extract_year_from_filename(file_path):
    """
    Extract the year from the Excel file name
    
    Args:
        file_path: Path to the Excel file
    
    Returns:
        int: Extracted year or current year if year cannot be extracted
    """
    try:
        # Extract filename from path and remove extension
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Try to find a 4-digit year in the filename
        year_match = re.search(r'20\d{2}', filename_no_ext)
        if year_match:
            return int(year_match.group(0))
        
        # If no year found, return current year as fallback
        return datetime.now().year
    except Exception as e:
        print(f"Error extracting year from filename: {e}")
        return datetime.now().year

def load_excel_file(file_path):
    """
    Load an Excel file and return the ExcelFile object
    
    Args:
        file_path: Path to the Excel file
    
    Returns:
        pandas.ExcelFile: Loaded Excel file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.ExcelFile(file_path)

def convert_shift_to_time(shift):
    """
    Convert shift number to time
    
    Args:
        shift: Shift number/string
    
    Returns:
        str: Time in HH:MM format
    """
    shift_mapping = {
        1: "06:00",
        2: "14:00",
        3: "22:00",
        "1": "06:00",
        "2": "14:00",
        "3": "22:00"
    }
    
    return shift_mapping.get(shift, "00:00")  # Default to 00:00 if shift not found

def get_month_number(sheet_name):
    """
    Get month number from sheet name
    
    Args:
        sheet_name: Name of the sheet (representing the month)
    
    Returns:
        int: Month number (1-12)
    """
    month_mapping = {
        "Януари": 1, "Февруари": 2, "Март": 3, "Април": 4,
        "Май": 5, "Юни": 6, "Юли": 7, "Август": 8,
        "Септември": 9, "Октомври": 10, "Ноември": 11, "Декември": 12
    }
    
    # Try to extract month from sheet name or use sheet name as a number
    try:
        if sheet_name in month_mapping:
            month = month_mapping[sheet_name]
        else:
            # Try to extract month number from sheet name if it's not a Bulgarian month name
            month = int(sheet_name)
    except (ValueError, TypeError):
        # Default to 1 if month cannot be determined
        month = 1
    
    # Ensure month is between 1-12
    return max(1, min(12, month))

def generate_dates_for_month(month, year):
    """
    Generate all dates for a given month and year
    
    Args:
        month: Month number (1-12)
        year: Year
    
    Returns:
        list: List of date strings in YYYY-MM-DD format
    """
    # Get number of days in month
    days_in_month = calendar.monthrange(year, month)[1]
    
    # If it's the current month, limit to current day
    current_date = datetime.now()
    if year == current_date.year and month == current_date.month:
        days_in_month = min(days_in_month, current_date.day)
    
    # Generate dates
    return [f"{year}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]

def process_sheet(sheet_data, sheet_name, year):
    """
    Process a single sheet from the Excel file
    
    Args:
        sheet_data: DataFrame containing sheet data
        sheet_name: Name of the sheet (representing the month)
        year: Year to use for timestamps
    
    Returns:
        DataFrame: Processed data with timestamp
    """
    # Skip if DataFrame is empty
    if sheet_data.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid SettingWithCopyWarning
    df = sheet_data.copy()
    
    # Clean the data - drop rows with "Общо" in "Смяна" column
    if "Смяна" in df.columns:
        df = df[~df["Смяна"].astype(str).str.contains("Общо")]
        
    # Get month number from sheet name
    month = get_month_number(sheet_name)
    
    # Create timestamp column
    if "Смяна" in df.columns:
        # Generate all dates for the month
        month_dates = generate_dates_for_month(month, year)
        
        if len(month_dates) == 0:
            # No valid dates for this month (might be a future month)
            return pd.DataFrame()
        
        # Create a new dataframe with proper timestamps
        result_rows = []
        
        # Expected shift numbers (could be strings or integers)
        expected_shifts = ['1', '2', '3'] if isinstance(df["Смяна"].iloc[0], str) else [1, 2, 3]
        
        # Group by unique shifts
        shift_groups = {}
        for shift in expected_shifts:
            shift_groups[shift] = df[df["Смяна"] == shift]
        
        # Determine the number of days we can fully cover
        min_shift_count = min(len(group) for group in shift_groups.values()) if shift_groups else 0
        
        # Ensure we don't exceed the days in the month
        day_count = min(min_shift_count, len(month_dates))
        
        # For each day, create entries for each shift
        for day_idx in range(day_count):
            current_date = month_dates[day_idx]
            
            # For each shift, get the corresponding row and update its timestamp
            for shift in expected_shifts:
                if day_idx < len(shift_groups[shift]):
                    row = shift_groups[shift].iloc[day_idx].copy()
                    
                    # Create timestamp
                    time_str = convert_shift_to_time(shift)
                    row["TimeStamp"] = pd.to_datetime(f"{current_date} {time_str}", errors='coerce')
                    
                    # Update the original date column with the formatted date
                    row["Дата"] = pd.to_datetime(current_date).strftime('%d.%m.%Y')
                    
                    result_rows.append(row)
        
        # Create a new dataframe from the processed rows
        if result_rows:
            return pd.DataFrame(result_rows)
        else:
            return pd.DataFrame()
    
    return df

def excel_to_dataframe(file_path):
    """
    Convert Excel file with multiple sheets to a single DataFrame
    
    Args:
        file_path: Path to the Excel file
    
    Returns:
        DataFrame: Combined data from all sheets with timestamp
    
    Example:
        ```python
        import pandas as pd
        from dispather_excel_processor import excel_to_dataframe
        
        # Process the Excel file
        file_path = "path/to/Doklad_Dispecheri_2024.xlsx"
        df = excel_to_dataframe(file_path)
        
        # Now you can work with the DataFrame
        print(df.head())
        
        # Save to file if needed
        df.to_csv("output.csv", index=False)
        ```
    """
    # Extract year from filename
    year = extract_year_from_filename(file_path)
    print(f"Extracted year from filename: {year}")
    
    excel_file = load_excel_file(file_path)
    
    # Get all sheet names
    sheet_names = excel_file.sheet_names
    
    # Process each sheet and collect the resulting DataFrames
    all_data = []
    for sheet_name in sheet_names:
        try:
            sheet_data = excel_file.parse(sheet_name)
            processed_data = process_sheet(sheet_data, sheet_name, year)
            
            if not processed_data.empty:
                # Add sheet name as a column for reference
                processed_data.loc[:, "OriginalSheet"] = sheet_name
                all_data.append(processed_data)
        except Exception as e:
            print(f"Error processing sheet '{sheet_name}': {e}")
    
    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    file_path = "dispathers/Doklad_Dispecheri_2024.xlsx"
    
    try:
        combined_data = excel_to_dataframe(file_path)
        print(f"Successfully processed Excel file. DataFrame shape: {combined_data.shape}")
        print("\nDataFrame columns:")
        print(combined_data.columns.tolist())
        
        # Print first few rows
        print("\nFirst few rows:")
        print(combined_data.head())
        
        # Save the result to CSV
        output_file = "processed_dispathers_data.csv"
        combined_data.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing Excel file: {e}")
