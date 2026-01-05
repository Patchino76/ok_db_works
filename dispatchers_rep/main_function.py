import sys
import os
from excel_to_pg import ExcelToPGConverter

def main():
    """
    Main function to run the Excel to PostgreSQL conversion.
    Processes both 2025 and 2026 Excel files.
    """
    # PostgreSQL connection parameters (from scheduled_pulse_postgresql.py)
    pg_host = 'em-m-db4.ellatzite-med.com'  # PostgreSQL server host
    pg_port = 5432                         # PostgreSQL server port
    pg_dbname = 'em_pulse_data'            # PostgreSQL database name
    pg_user = 's.lyubenov'                 # PostgreSQL username
    pg_password = 'tP9uB7sH7mK6zA7t'      # PostgreSQL password
    
    # Create converter
    converter = ExcelToPGConverter(
        pg_host=pg_host,
        pg_port=pg_port,
        pg_dbname=pg_dbname,
        pg_user=pg_user,
        pg_password=pg_password
    )
    
    # Define files to process in order
    files_to_process = []
    
    # Check for 2025 file
    if os.path.exists('Doklad_Dispecheri_2025!.xlsx'):
        files_to_process.append(('Doklad_Dispecheri_2025!.xlsx', False))  # (file, append_mode)
    elif os.path.exists(os.path.join('dispatchers_rep', 'Doklad_Dispecheri_2025!.xlsx')):
        files_to_process.append((os.path.join('dispatchers_rep', 'Doklad_Dispecheri_2025!.xlsx'), False))
    
    # Check for 2026 file
    if os.path.exists('Doklad_Dispecheri_2026!.xlsx'):
        files_to_process.append(('Doklad_Dispecheri_2026!.xlsx', True))  # Append mode for 2026
    elif os.path.exists(os.path.join('dispatchers_rep', 'Doklad_Dispecheri_2026!.xlsx')):
        files_to_process.append((os.path.join('dispatchers_rep', 'Doklad_Dispecheri_2026!.xlsx'), True))
    
    # Check if we have files to process
    if not files_to_process:
        print("Error: Could not find any Excel files (2025 or 2026).")
        sys.exit(1)
    
    # Process each file
    all_success = True
    for input_file, append_mode in files_to_process:
        # Verify file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            all_success = False
            continue
            
        mode_str = "APPEND" if append_mode else "CREATE"
        print(f"\n{'='*60}")
        print(f"Processing file [{mode_str} MODE]: {input_file}")
        print(f"{'='*60}\n")
        
        # Process the data and insert into PostgreSQL
        success = converter.process_excel_to_pg(input_file, append_mode=append_mode)
        
        if success:
            print(f"\n✅ Successfully processed: {input_file}")
        else:
            print(f"\n❌ Failed to process: {input_file}")
            all_success = False
    
    # Final summary
    print(f"\n{'='*60}")
    if all_success:
        print("✅ All Excel files were successfully processed and inserted into PostgreSQL")
    else:
        print("⚠️ Some files failed to process. Check logs above for details.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
