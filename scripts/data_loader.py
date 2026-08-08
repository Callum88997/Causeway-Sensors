import os
import pandas as pd

DATA_DIR = "data"
TABLE_DIR = os.path.join(DATA_DIR, "tables")
BUCKET_DIR = os.path.join(DATA_DIR, "buckets")

time_cols = ['t', 'time']

def convert_numeric_columns(df, threshold=0.9):
    """
    Convert numeric-looking values to floats while preserving text.
    """
    for col in df.columns:

        converted = pd.to_numeric(df[col], errors="coerce")
        numeric_ratio = converted.notna().sum() / len(df)

        if numeric_ratio >= threshold:
            df[col] = converted

    return df

def is_valid_prg_only(df, file_name):
    """
    Returns True if 'prg' is the only active reagent, False otherwise.
    Ignores standard control steps like buffers, baselines, and start/finish markers.
    """
    info_col = next((col for col in ['information'] if col in df.columns), None)
    
    if info_col:

        ignore_pattern = 'buffer|baseline|start|finish|initial|final|concentration|plateau'

        reagent_rows = df[~df[info_col].astype(str).str.contains(ignore_pattern, case=False, na=False)]
        
        non_prg_rows = reagent_rows[~reagent_rows[info_col].astype(str).str.contains('PrG', case=False, na=False)]
        
        if not non_prg_rows.empty:

            violating_stages = non_prg_rows[info_col].unique().tolist()
            print(f"SKIPPING: Flag file '{file_name}' contains non-'prg' reagents: {violating_stages}")

            return False
            
    return True

# Load all database tables
def load_tables():
    tables = {}

    for file in os.listdir(TABLE_DIR):

        if file.endswith(".csv"):

            name = file.replace(".csv", "")
            path = os.path.join(TABLE_DIR, file)
            print(f"Loading table: {name}")

            try:

                df = pd.read_csv(path)
                df = convert_numeric_columns(df)
                tables[name] = df
            except pd.errors.EmptyDataError:

                print(f"EMPTY FILE SKIPPED: {file}")
                tables[name] = pd.DataFrame()
            
    return tables

# Load bucket files recursively, taking tables to filter refPoly4
def load_bucket_files(tables):
    buckets = {}
    
    if not os.path.exists(BUCKET_DIR):
        print(f"Directory not found: {BUCKET_DIR}")
        return buckets

    bucket_dirs = [b for b in os.listdir(BUCKET_DIR) if os.path.isdir(os.path.join(BUCKET_DIR, b))]
    
    bucket_dirs.sort(key=lambda x: 0 if 'amf_flags' in x.lower() else 1)

    # 3. Create a lookup dictionary mapping refPoly4 base filenames to amf_flags base filenames
    ref_to_amf_map = {}
    if 'immob' in tables and not tables['immob'].empty:
        immob_df = tables['immob']
        if 'refPoly4' in immob_df.columns and 'amfFlags' in immob_df.columns:
            for _, row in immob_df.iterrows():
                
                # Extract just the filename without extensions for reliable matching
                ref_base = os.path.splitext(os.path.basename(str(row['refPoly4'])))[0]
                amf_base = os.path.splitext(os.path.basename(str(row['amfFlags'])))[0]
                ref_to_amf_map[ref_base] = amf_base

    # Track which amf_flags files successfully passed the PrG-only check
    valid_amf_basenames = set()

    for bucket in bucket_dirs:
        bucket_path = os.path.join(BUCKET_DIR, bucket)
        buckets[bucket] = {}
        
        is_flag_bucket = 'amf_flags' in bucket.lower()
        is_refpoly_bucket = 'refpoly4' in bucket.lower()

        for root, dirs, files in os.walk(bucket_path):
            for file in files:
                path = os.path.join(root, file)
                relative_path = os.path.relpath(path, bucket_path)
                file_basename = os.path.splitext(file)[0]

                # 4. If we are in the refPoly4 bucket, check our valid mappings
                if is_refpoly_bucket and ref_to_amf_map:
                    corresponding_amf = ref_to_amf_map.get(file_basename)
                    
                    if not corresponding_amf or corresponding_amf not in valid_amf_basenames:
                        print(f"SKIPPING REFPOLY: '{file}' (mapped AMF_FLAGS '{corresponding_amf}' is missing or invalid)")
                        continue

                try:
                    if file.endswith(".parquet"):
                        df = pd.read_parquet(path)
                    else:
                        df = pd.read_csv(path)
                    
                    df = convert_numeric_columns(df)

                    for t in time_cols:
                        if t in df.columns:
                            df = df.sort_values(t)

                    if is_flag_bucket:
                        if "time" in df.columns:
                            time_str = df["time"].astype(str)
                            if time_str.str.contains(r"[:-]").any():
                                t_td = pd.to_timedelta(time_str.str.replace("-", ":", regex=False))
                                start = t_td.iloc[0].floor("min")
                                df["time"] = (t_td - start).dt.total_seconds().astype(int)

                        # Run validation check
                        if not is_valid_prg_only(df, file):
                            continue
                        
                        # Add successfully validated amf_flags file to our tracking set
                        valid_amf_basenames.add(file_basename)

                    buckets[bucket][relative_path] = df

                except Exception as e:
                    print(f"Failed loading {path}: {e}")
                    
    return buckets

def load_titan_data():
    """
    Main orchestration method to load all Titan data.
    It loads the structural tables first, then uses them to selectively
    load and filter the bucket datasets.
    """
    print("Loading database tables...")
    tables = load_tables()
    
    print("Loading and filtering bucket data...")
    buckets = load_bucket_files(tables)
    
    print("Titan data load complete.")
    return tables, buckets