"""
Translate DataFrame Columns Script

This script:
1. Opens the processed_dispatcher_data_2024.csv file
2. Creates a DataFrame from it
3. Translates column names from Bulgarian to English using columns_dict
4. Removes any columns that don't have a mapping in the dictionary
5. Saves the result to a new CSV file with English column names
"""

import pandas as pd
from columns_dictionary import columns_dict

def translate_and_filter_columns():
    """
    Open CSV file, translate column names, filter columns, and save to new file
    """
    # Define the input and output file paths
    input_file = "processed_dispatcher_data_2025.csv"
    output_file = "processed_dispatcher_data_2025_en.csv"
    
    print(f"Processing file: {input_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"Original DataFrame shape: {df.shape}")
        print(f"Original columns: {len(df.columns)}")
        
        # Create a Bulgarian to English mapping 
        bg_to_en = {bg: en for en, bg in columns_dict.items()}
        
        # Add 'OriginalSheet' to 'original_sheet' mapping if it exists
        if 'OriginalSheet' in df.columns and 'original_sheet' not in bg_to_en:
            bg_to_en['OriginalSheet'] = 'original_sheet'
        
        # Print first few Bulgarian columns
        print("\nFirst 5 original column names:")
        for col in list(df.columns)[:5]:
            print(f"  - {col}")
        
        # Identify which columns are in our dictionary
        columns_to_keep = []
        column_mapping = {}
        
        for col in df.columns:
            # Check exact matches
            if col in bg_to_en:
                columns_to_keep.append(col)
                column_mapping[col] = bg_to_en[col]
                continue
                
            # Check for columns with newlines or extra spaces
            for bg_col in bg_to_en.keys():
                # Strip spaces, newlines, and tabs for comparison
                clean_col = col.replace('\n', '').replace('\r', '').strip()
                clean_bg_col = bg_col.replace('\n', '').replace('\r', '').strip()
                
                if clean_col == clean_bg_col:
                    columns_to_keep.append(col)
                    column_mapping[col] = bg_to_en[bg_col]
                    break
                    
        # Keep only the columns we found mappings for
        df_filtered = df[columns_to_keep].copy()
        
        # Rename the columns using our mapping
        df_filtered = df_filtered.rename(columns=column_mapping)
        
        print(f"\nAfter filtering, kept {len(columns_to_keep)} out of {len(df.columns)} columns")
        print(f"New DataFrame shape: {df_filtered.shape}")
        
        # Print first few English columns
        print("\nFirst 5 translated column names:")
        for col in list(df_filtered.columns)[:5]:
            print(f"  - {col}")
        
        # Print mapping for debugging
        print("\nColumn mapping used:")
        for bg, en in column_mapping.items():
            print(f"  - '{bg}' → '{en}'")
        
        # Save to new CSV file
        df_filtered.to_csv(output_file, index=False)
        print(f"\nSaved translated DataFrame to: {output_file}")
        
        return df_filtered
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

if __name__ == "__main__":
    translate_and_filter_columns()
