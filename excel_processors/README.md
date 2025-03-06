# Excel Dispatcher Data Processor

This tool processes the dispatcher Excel file `Doklad_Dispecheri_YYYY.xlsx` and converts it into a pandas DataFrame with proper timestamps.

## Features

- Loads data from all 12 monthly sheets
- Automatically extracts the year from the Excel file name (e.g., `Doklad_Dispecheri_2025.xlsx` uses 2025 as the year)
- Creates proper timestamps for each row with:
  - Year: Extracted from the filename
  - Month: Based on sheet name
  - Day: Based on row sequence or existing date data
  - Time: Based on shift (Shift 1: 06:00, Shift 2: 14:00, Shift 3: 22:00)
- All three shifts (1, 2, 3) occur on the same day (e.g., 2025-01-01 for Shift 1 at 06:00, Shift 2 at 14:00, and Shift 3 at 22:00)
- Filters out rows containing "Общо" in the "Смяна" column
- Preserves all original columns from the Excel file
- Adds the sheet name as a reference column
- Handles current month intelligently to avoid generating entries for non-existent days

## Usage

### Simple Usage

Run the `convert_dispatcher_data.py` script to process the file and save the results:

```bash
python convert_dispatcher_data.py
```

This will:
1. Load the Excel file from the `dispathers` directory
2. Extract the year from the filename
3. Process all sheets
4. Save the output as both CSV and Excel files in the current directory
5. Display various statistics and data quality checks

### API Usage

You can also use the functions directly in your own code:

```python
from dispather_excel_processor import excel_to_dataframe

# Process the Excel file
file_path = "path/to/Doklad_Dispecheri_2025.xlsx"
df = excel_to_dataframe(file_path)

# Now you can work with the DataFrame
print(df.head())

# Save to file if needed
df.to_csv("output.csv", index=False)
```

## Output

The script generates the following outputs:

1. **processed_dispatcher_data_YYYY.csv** - CSV file containing the processed data
2. **processed_dispatcher_data_YYYY.xlsx** - Excel file containing the processed data

Where YYYY is the year extracted from the input file name.

The resulting DataFrame includes:

- All original columns from the Excel file
- A `TimeStamp` column with proper datetime values (YYYY-MM-DD HH:MM:SS format)
- An `OriginalSheet` column indicating the source sheet

## Data Structure

The output DataFrame has the following key columns:

- `Дата` - Date in DD.MM.YYYY format
- `Смяна` - Shift number (1, 2, or 3)
- `TimeStamp` - Proper datetime timestamp combining date and shift time
- `OriginalSheet` - Source sheet name (month)

## Data Quality Checks

The script performs various data quality checks including:

- Checking for missing values in critical columns
- Verifying shift distribution is balanced
- Displaying timestamp statistics (earliest/latest dates, number of unique dates)
- Showing sample timestamps for each month and shift

## Requirements

- Python 3.6+
- pandas
- openpyxl (for Excel file handling)

## Notes

- The script automatically extracts the year from the Excel filename
- When no year can be extracted from the filename, the current year is used as a fallback
- When processing the current month, the script will only generate entries up to the current day to avoid creating data for non-existent days
