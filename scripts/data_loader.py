# Imports required packages
import os
import pandas as pd
import re

# Defines configuration constants for data directories
DATA_DIR = 'data'
TABLE_DIR = os.path.join(DATA_DIR, 'tables')
BUCKET_DIR = os.path.join(DATA_DIR, 'buckets')

# Declares column names representing time
time_cols = ['t', 'time']

def convert_numeric_columns(df, threshold=0.9):
    '''Converts numeric-looking values in a dataframe to floats while preserving text.

    Args:
        df (pd.DataFrame): The dataframe to process.
        threshold (float): The minimum ratio of valid numeric values required to convert a column.

    Returns:
        pd.DataFrame: The processed dataframe with converted columns.
    '''

    # Loops through each column in the dataframe
    for col in df.columns:

        # Attempts to convert column values to numeric, coercing errors to NaN
        converted = pd.to_numeric(df[col], errors='coerce')

        # Calculates the ratio of valid numeric values
        numeric_ratio = converted.notna().sum() / len(df)

        # Updates the column if the numeric ratio meets or exceeds the threshold
        if numeric_ratio >= threshold:
            df[col] = converted

    # Returns the updated dataframe
    return df

def is_valid_chip(file_name):
    '''Evaluates if the file name contains a valid chip ID.

    Args:
        file_name (str): The name of the file being validated.

    Returns:
        bool: True if the file name contains a valid chip format, False otherwise.
    '''

    # Defines the regex pattern for a real chip
    # Looks for 'B' or 'R' followed by digits then 'R' followed by digits
    chip_pattern = r'B\d+R\d+|R\d+R\d+'
    
    # Returns True if the pattern is found in the filename
    return bool(re.search(chip_pattern, file_name))

def is_valid_prg_only(df, file_name):
    '''Evaluates if 'prg' is the only active reagent, ignoring standard control steps.

    Args:
        df (pd.DataFrame): The dataframe containing the flag data.
        file_name (str): The name of the file being validated.

    Returns:
        bool: True if 'prg' is the only active reagent, False otherwise.
    '''

    # Identifies the information column if it exists in the dataframe
    info_col = next((col for col in ['information'] if col in df.columns), None)
    
    # Processes the dataframe if the information column is found
    if info_col:

        # Defines the regex pattern for standard control steps to ignore
        ignore_pattern = 'buffer|baseline|start|finish|initial|final|concentration|plateau'

        # Filters out rows that match the ignore pattern
        reagent_rows = df[~df[info_col].astype(str).str.contains(ignore_pattern, case=False, na=False)]
        
        # Identifies rows that contain reagents other than 'PrG'
        non_prg_rows = reagent_rows[~reagent_rows[info_col].astype(str).str.contains('PrG', case=False, na=False)]
        
        # Checks whether the non-prg dataframe contains data
        if not non_prg_rows.empty:

            # Extracts a list of the violating stages
            violating_stages = non_prg_rows[info_col].unique().tolist()

            # Prints a skipping warning with the violating stages
            print(f"SKIPPING: Flag file '{file_name}' contains non-'prg' reagents: {violating_stages}")

            # Returns False indicating validation failure
            return False
            
    # Returns True if validation passes
    return True

def load_tables():
    '''Loads exported database tables into dataframes keyed by table name.

    Returns:
        dict: Mapping of table names to their loaded pandas dataframes.
    '''

    # Initialises an empty dictionary to store the tables
    tables = {}

    # Loops through each file in the tables directory
    for file in os.listdir(TABLE_DIR):

        # Processes only CSV files
        if file.endswith('.csv'):

            # Extracts the table name by removing the file extension
            name = file.replace('.csv', '')

            # Constructs the full file path
            path = os.path.join(TABLE_DIR, file)

            # Prints the current table being loaded
            print(f'Loading table: {name}')

            # Attempts to load and process the table data
            try:

                # Reads the CSV file into a dataframe
                df = pd.read_csv(path)

                # Converts applicable columns to numeric types
                df = convert_numeric_columns(df)

                # Assigns the processed dataframe to the dictionary
                tables[name] = df

            # Catches empty data errors and stores an empty dataframe
            except pd.errors.EmptyDataError:

                # Prints a warning for empty files
                print(f'EMPTY FILE SKIPPED: {file}')

                # Assigns an empty dataframe
                tables[name] = pd.DataFrame()
            
    # Returns the dictionary of loaded tables
    return tables

def load_bucket_files(tables):
    '''Loads and organises bucket files required for Titan analysis.

    Args:
        tables (dict): Database tables used to identify and validate file records.

    Returns:
        dict: Nested mapping of bucket names to their loaded dataframes.
    '''

    # Initialises an empty dictionary for the buckets
    buckets = {}
    
    # Checks if the bucket directory exists and returns an empty dictionary if not
    if not os.path.exists(BUCKET_DIR):
        print(f'Directory not found: {BUCKET_DIR}')
        return buckets

    # Retrieves a list of all subdirectories within the bucket directory
    bucket_dirs = [b for b in os.listdir(BUCKET_DIR) if os.path.isdir(os.path.join(BUCKET_DIR, b))]
    
    # Sorts the directories to prioritise 'amf_flags'
    bucket_dirs.sort(key=lambda x: 0 if 'amf_flags' in x.lower() else 1)

    # Creates a lookup dictionary mapping refPoly4 base filenames to amf_flags base filenames
    ref_to_amf_map = {}

    # Checks whether the immob dataframe contains data
    if 'immob' in tables and not tables['immob'].empty:

        # Assigns the immob dataframe to a variable
        immob_df = tables['immob']

        # Verifies that required columns exist in the dataframe
        if 'refPoly4' in immob_df.columns and 'amfFlags' in immob_df.columns:

            # Loops through each row in the dataframe
            for _, row in immob_df.iterrows():
                
                # Extracts just the filename without extensions for reliable matching
                ref_base = os.path.splitext(os.path.basename(str(row['refPoly4'])))[0]
                amf_base = os.path.splitext(os.path.basename(str(row['amfFlags'])))[0]

                # Maps the refPoly4 base name to the amfFlags base name
                ref_to_amf_map[ref_base] = amf_base

    # Tracks which amf_flags files successfully passed the PrG-only check
    valid_amf_basenames = set()

    # Loops through each bucket directory
    for bucket in bucket_dirs:

        # Constructs the full bucket path
        bucket_path = os.path.join(BUCKET_DIR, bucket)

        # Initialises an empty dictionary for the current bucket
        buckets[bucket] = {}
        
        # Sets boolean flags to identify specific bucket types
        is_flag_bucket = 'amf_flags' in bucket.lower()
        is_refpoly_bucket = 'refpoly4' in bucket.lower()

        # Traverses the directory tree within the current bucket
        for root, dirs, files in os.walk(bucket_path):

            # Loops through each file in the current directory
            for file in files:

                # Validates the file name to ensure it contains a valid chip ID
                if not is_valid_chip(file):
                    print(f"SKIPPING: '{file}' (Does not contain a valid chip ID)")
                    continue

                # Constructs paths and extracts the base filename
                path = os.path.join(root, file)
                relative_path = os.path.relpath(path, bucket_path)
                file_basename = os.path.splitext(file)[0]

                # Checks valid mappings if processing the refPoly4 bucket
                if is_refpoly_bucket and ref_to_amf_map:

                    # Retrieves the corresponding amf_flags filename
                    corresponding_amf = ref_to_amf_map.get(file_basename)
                    
                    # Skips the file if the mapping is missing or invalid
                    if not corresponding_amf or corresponding_amf not in valid_amf_basenames:
                        print(f"SKIPPING REFPOLY: '{file}' (mapped AMF_FLAGS '{corresponding_amf}' is missing or invalid)")
                        continue

                # Attempts to load and process the file data
                try:

                    # Reads parquet files
                    if file.endswith('.parquet'):
                        df = pd.read_parquet(path)

                    # Reads standard CSV files
                    else:
                        df = pd.read_csv(path)
                    
                    # Converts applicable columns to numeric types
                    df = convert_numeric_columns(df)

                    # Loops through predefined time columns to sort the data
                    for t in time_cols:
                        if t in df.columns:
                            df = df.sort_values(t)

                    # Normalises the time format for flag buckets
                    if is_flag_bucket:

                        # Checks if the time column exists
                        if 'time' in df.columns:

                            # Converts the time column to strings
                            time_str = df['time'].astype(str)

                            # Formats and calculates elapsed seconds if time strings contain separators
                            if time_str.str.contains(r'[:-]').any():

                                t_td = pd.to_timedelta(time_str.str.replace('-', ':', regex=False))
                                start = t_td.iloc[0].floor('min')
                                
                                df['time'] = (t_td - start).dt.total_seconds().astype(int)

                        # Runs the validation check to ensure only 'prg' is active
                        if not is_valid_prg_only(df, file):
                            continue
                        
                        # Adds successfully validated amf_flags files to the tracking set
                        valid_amf_basenames.add(file_basename)

                    # Stores the processed dataframe in the buckets dictionary
                    buckets[bucket][relative_path] = df

                # Catches and reports any errors during file loading
                except Exception as e:
                    print(f'Failed loading {path}: {e}')
                    
    # Returns the nested dictionary of loaded bucket files
    return buckets

def load_titan_data():
    '''Orchestrates the data loading process for Titan analysis.
    
    Loads the structural database tables first, then uses them to selectively
    load and filter the bucket datasets.

    Returns:
        tuple: A tuple containing the tables dictionary and buckets dictionary.
    '''

    # Prints a progress message for table loading
    print('Loading database tables...')

    # Loads the database tables
    tables = load_tables()
    
    # Prints a progress message for bucket loading
    print('Loading and filtering bucket data...')

    # Loads and organises the bucket files
    buckets = load_bucket_files(tables)
    
    # Prints a completion message
    print('Titan data load complete.')

    # Returns the loaded tables and buckets
    return tables, buckets
