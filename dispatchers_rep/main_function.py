import sys
import os
from excel_to_pg import ExcelToPGConverter

def main():
    """
    Main function to run the Excel to PostgreSQL conversion.
    """
    # Check command line arguments for input file
    input_file = None
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # PostgreSQL connection parameters (from scheduled_pulse_postgresql.py)
    pg_host = 'em-m-db4.ellatzite-med.com'  # PostgreSQL server host
    pg_port = 5432                         # PostgreSQL server port
    pg_dbname = 'em_pulse_data'            # PostgreSQL database name
    pg_user = 's.lyubenov'                 # PostgreSQL username
    pg_password = 'tP9uB7sH7mK6zA7t'      # PostgreSQL password
    
    # If input file not provided via command line, use default paths
    if not input_file:
        # First check if file exists in current directory
        if os.path.exists('Doklad_Dispecheri_2025.xlsx'):
            input_file = 'Doklad_Dispecheri_2025.xlsx'
        # Then check in dispatchers_rep subdirectory
        elif os.path.exists(os.path.join('dispatchers_rep', 'Doklad_Dispecheri_2025.xlsx')):
            input_file = os.path.join('dispatchers_rep', 'Doklad_Dispecheri_2025.xlsx')
        # Check for filename with year pattern
        elif os.path.exists('Doklad_Dispecheri_2024.xlsx'):
            input_file = 'Doklad_Dispecheri_2024.xlsx'
        else:
            print("Error: Could not find Excel file. Please provide path as command line argument.")
            sys.exit(1)
    
    # Verify file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
        
    print(f"Using input file: {input_file}")
    
    # Create converter and process Excel
    converter = ExcelToPGConverter(
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
        pg_user=pg_user,
        pg_password=pg_password
    )
    
    # Process the data and insert into PostgreSQL
    success = converter.process_excel_to_pg(input_file)
    
    if success:
        print("\nExcel data was successfully processed and inserted into PostgreSQL")
    else:
        print("\nFailed to process Excel data or insert into PostgreSQL")


if __name__ == "__main__":
    main()
