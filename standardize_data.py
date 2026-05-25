import pandas as pd
import numpy as np
import os
import difflib

def standardize_project_data(input_file):
    """
    Standardizes an input CSV or Excel file into the target schema.
    
    Target Schema:
    Task_ID, Dependencies, Task_Name, Resource_Type, Unit_Value_EUR, 
    Unit_of_Measure, Quantity_Needed, Total_Cost, Crashability, Crash_Price_Increase_Pct
    """
    
    # 1. Read the input file
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return None

    file_ext = os.path.splitext(input_file)[1].lower()
    try:
        if file_ext == '.csv':
            # Try different encodings for CSV
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(input_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV with common encodings.")
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return None

    if df.empty:
        print(f"Warning: The file {input_file} is empty.")
        return df

    # 2. Define target schema and aliases
    target_columns = [
        "Task_ID", "Dependencies", "Task_Name", "Resource_Type", 
        "Unit_Value_EUR", "Unit_of_Measure", "Quantity_Needed", 
        "Total_Cost", "Crashability", "Crash_Price_Increase_Pct"
    ]
    
    aliases = {
        "Task_ID": ["ID", "Task ID", "TaskID", "WBS", "Codice", "Task No", "Task#"],
        "Dependencies": ["Predecessors", "Depends on", "Dep", "Dipendenze", "Predecessori", "Succ", "Pred"],
        "Task_Name": ["Name", "Task Name", "Description", "Descrizione", "Nome Task", "Attività", "Oggetto"],
        "Resource_Type": ["Resource", "Type", "Resource Type", "Tipo Risorsa", "Categoria", "Risorsa"],
        "Unit_Value_EUR": ["Rate", "Unit Cost", "Unit Price", "Unit Value", "Prezzo Unitario", "Valore Unitario", "PU", "Tariffa"],
        "Unit_of_Measure": ["UOM", "Unit", "Measure", "Unità di misura", "UM", "Unità", "Misura"],
        "Quantity_Needed": ["Qty", "Quantity", "Amount", "Quantità", "Q.tà", "Volume"],
        "Total_Cost": ["Cost", "Total", "Price", "Importo", "Costo Totale", "Totale", "Costo"],
        "Crashability": ["Crashable", "Can Crash", "Crashabilità", "Riducibile", "Accelerabile"],
        "Crash_Price_Increase_Pct": ["Crash Pct", "Crash Increase", "Crash Price %", "Incremento Crash %", "Sovrapprezzo Crash", "Crash %"]
    }

    # 3. Align columns using fuzzy matching and aliases
    current_columns = df.columns.tolist()
    column_mapping = {}
    
    for target in target_columns:
        # Check if target name exists exactly (case-insensitive)
        exact_match = next((c for c in current_columns if c.lower() == target.lower()), None)
        if exact_match:
            column_mapping[exact_match] = target
            continue
            
        # Check aliases (case-insensitive)
        found = False
        for alias in aliases.get(target, []):
            alias_match = next((c for c in current_columns if c.lower() == alias.lower()), None)
            if alias_match:
                column_mapping[alias_match] = target
                found = True
                break
        if found:
            continue
            
        # Fuzzy matching as fallback
        matches = difflib.get_close_matches(target, current_columns, n=1, cutoff=0.6)
        if matches:
            column_mapping[matches[0]] = target
        else:
            # Try fuzzy matching on aliases too
            for alias in aliases.get(target, []):
                alias_matches = difflib.get_close_matches(alias, current_columns, n=1, cutoff=0.7)
                if alias_matches:
                    column_mapping[alias_matches[0]] = target
                    found = True
                    break
            if not found:
                print(f"Warning: Could not find a match for required column '{target}'")

    # Rename columns based on mapping
    df = df.rename(columns=column_mapping)
    
    # 4. Ensure all target columns exist (fill with NaN if missing)
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Select only target columns in the correct order
    df = df[target_columns]

    # 5. Data Cleaning
    
    # Clean Resource_Type
    if 'Resource_Type' in df.columns:
        df['Resource_Type'] = df['Resource_Type'].astype(str).str.strip()
        resource_map = {
            'Labor': 'Cost of Labour',
            'Labour': 'Cost of Labour',
            'Manodopera': 'Cost of Labour',
            'Lavoro': 'Cost of Labour',
            'Material': 'Cost of Material',
            'Materials': 'Cost of Material',
            'Materiale': 'Cost of Material',
            'Materiali': 'Cost of Material',
            'Noleggio': 'Cost of Equipment',
            'Equipment': 'Cost of Equipment',
            'Attrezzatura': 'Cost of Equipment'
        }
        # Case-insensitive replacement
        for k, v in resource_map.items():
            df.loc[df['Resource_Type'].str.lower() == k.lower(), 'Resource_Type'] = v

    # Parse percentages in Crash_Price_Increase_Pct to floats
    if 'Crash_Price_Increase_Pct' in df.columns:
        def parse_pct(val):
            if pd.isna(val) or val == '' or str(val).lower() == 'nan':
                return 0.0
            
            s_val = str(val).strip()
            has_percent = '%' in s_val
            
            # Remove % and handle European comma decimal separator
            s_val = s_val.replace('%', '').replace(',', '.').strip()
            
            try:
                f_val = float(s_val)
                # If it had a '%' sign, OR if the number is > 1.0 (e.g., 35), 
                # we assume it's a percentage and divide by 100.
                # If it's already a decimal like 0.35, we keep it as is.
                if has_percent or f_val > 1.0:
                    return f_val / 100.0
                return f_val
            except ValueError:
                return 0.0

        df['Crash_Price_Increase_Pct'] = df['Crash_Price_Increase_Pct'].apply(parse_pct)

    # Convert numeric columns
    numeric_cols = ['Unit_Value_EUR', 'Quantity_Needed', 'Total_Cost', 'Crash_Price_Increase_Pct']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '.').replace('nan', '0')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Clean Dependencies
    df['Dependencies'] = df['Dependencies'].fillna("").astype(str).str.replace(';', ',')

    # 6. Output and save
    output_filename = "standardized_project_dataset.csv"
    try:
        df.to_csv(output_filename, index=False)
        print(f"Standardized data saved to {output_filename}")
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")
    
    return df

if __name__ == "__main__":
    # Test with one of the existing datasets if available
    test_file = "PCM - Dataset (.csv)/Exp.3 - randomised_crashing_dataset.csv"
    if os.path.exists(test_file):
        print(f"Testing with {test_file}...")
        standardize_project_data(test_file)
    else:
        print("Test file not found. Please provide an input file to standardize.")
