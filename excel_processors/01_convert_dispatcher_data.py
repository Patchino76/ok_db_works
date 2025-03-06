import os
import pandas as pd
from dispather_excel_processor import excel_to_dataframe, extract_year_from_filename

def main():
    """
    Main function to demonstrate how to use the excel_to_dataframe function
    """
    # Set the path to the Excel file
    file_path = os.path.join("xls_data", "Doklad_Dispecheri_2024.xlsx")
    
    # Process the Excel file
    print(f"Processing Excel file: {file_path}")
    df = excel_to_dataframe(file_path)
    
    # Show information about the resulting DataFrame
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show the first few rows
    print("\nFirst 5 rows:")
    print(df.head(5))
    
    # Show data types
    print("\nData types:")
    print(df.dtypes)
    
    # Data quality checks
    print("\nData Quality Checks:")
    
    # Check for missing values in critical columns
    critical_columns = ['Дата', 'Смяна', 'TimeStamp']
    for col in critical_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"Missing values in {col}: {missing} ({missing/len(df):.2%})")
    
    # Check shift distribution
    if 'Смяна' in df.columns:
        print("\nShift distribution:")
        shift_dist = df['Смяна'].value_counts().sort_index()
        print(shift_dist)
        
        # Check if shifts are balanced
        if len(shift_dist) > 0:
            min_count = shift_dist.min()
            max_count = shift_dist.max()
            if min_count == max_count:
                print("All shifts have the same number of entries: Balanced")
            else:
                print(f"Shift imbalance detected: Min = {min_count}, Max = {max_count}")
    
    # Get the year from the filename to use in the output filenames
    year = extract_year_from_filename(file_path)
    
    # Save to CSV file
    output_csv = f"processed_dispatcher_data_{year}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nDataFrame saved to: {output_csv}")
    
    # Save to Excel file
    # output_excel = f"processed_dispatcher_data_{year}.xlsx"
    # df.to_excel(output_excel, index=False)
    # print(f"DataFrame saved to: {output_excel}")
    
    # Show timestamp statistics
    if 'TimeStamp' in df.columns:
        print("\nTimestamp statistics:")
        print(f"Earliest date: {df['TimeStamp'].min()}")
        print(f"Latest date: {df['TimeStamp'].max()}")
        print(f"Number of unique dates: {df['TimeStamp'].dt.date.nunique()}")
        
        # Display sample of timestamps for each month
        print("\nSample timestamps by month:")
        for month in range(1, 13):
            month_data = df[df['TimeStamp'].dt.month == month]
            if not month_data.empty:
                # Get samples for each shift in this month
                for shift in sorted(month_data['Смяна'].unique()):
                    shift_data = month_data[month_data['Смяна'] == shift].head(1)
                    if not shift_data.empty:
                        print(f"Month {month}, Shift {shift}: {shift_data['TimeStamp'].iloc[0]}")
    
    # Generate some basic statistics on how many rows per month
    if 'OriginalSheet' in df.columns:
        print("\nRows per month:")
        month_stats = df['OriginalSheet'].value_counts()
        for month, count in month_stats.items():
            print(f"{month}: {count} rows")
    
    # Example of filtering the data
    print("\nExample: Filtering data for January - First month")
    # Get a sample month name from the first row of OriginalSheet
    if not df.empty and 'OriginalSheet' in df.columns:
        first_month = None
        month_sheet_names = df['OriginalSheet'].unique()
        if len(month_sheet_names) > 0:
            first_month = month_sheet_names[0]  # Get the first month name
        
        if first_month:
            print(f"Using month: {first_month}")
            january_data = df[df['OriginalSheet'] == first_month]
            print(f"Found {len(january_data)} rows for {first_month}")
            
            # Now filter for shift 1
            shift1_data = january_data[january_data['Смяна'] == 1]
            print(f"Found {len(shift1_data)} rows for {first_month}, Shift 1")
            if not shift1_data.empty:
                print(shift1_data[['Дата', 'Смяна', 'TimeStamp', 'OriginalSheet']].head(3))

if __name__ == "__main__":
    main()
