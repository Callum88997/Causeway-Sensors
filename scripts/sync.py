# Imports required packages
import os
import shutil
import pandas as pd
from scripts.supabase_client import supabase
from pandas.errors import EmptyDataError
from concurrent.futures import ThreadPoolExecutor

# Defines configuration constants for data directories
DATA_DIR = 'data'
TABLE_DIR = os.path.join(DATA_DIR, 'tables')
BUCKET_DIR = os.path.join(DATA_DIR, 'buckets')

# Threading configurations (Adjust if you hit rate limits)
MAX_TABLE_WORKERS = 5
MAX_FILE_WORKERS = 15

def get_tables():
    '''Retrieves the names of all accessible Supabase database tables.'''
    response = supabase.rpc('get_tables').execute()
    return [row['table_name'] for row in response.data]

def get_buckets():
    '''Retrieves the names of all accessible Supabase storage buckets.'''
    response = supabase.storage.list_buckets()
    return [bucket.name for bucket in response]

# Retrieves and stores the available tables and buckets
TABLES = get_tables()
BUCKETS = get_buckets()

def clear_local_data():
    '''Removes the previous local export and recreates the required directories.'''
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(TABLE_DIR)
    os.makedirs(BUCKET_DIR)
    print('Old data cleared')

def list_all_items(bucket, remote_path):
    '''Retrieves every item in a storage path, including paginated results.'''
    all_items = []
    offset = 0
    limit = 100

    while True:
        items = supabase.storage.from_(bucket).list(remote_path, {'limit': limit, 'offset': offset,})
        if not items:
            break
        all_items.extend(items)
        if len(items) < limit:
            break
        offset += limit
    return all_items

def process_single_table(table):
    '''Worker function to download and save a single table.'''
    print(f'Downloading table: {table}')
    response = supabase.table(table).select('*').execute()
    df = pd.DataFrame(response.data)
    filepath = os.path.join(TABLE_DIR, f'{table}.csv')
    df.to_csv(filepath, index=False)
    print(f'Saved {len(df)} rows for {table}')

def download_tables():
    '''Downloads configured database tables in parallel.'''
    print('\n--- Starting Table Downloads ---')
    with ThreadPoolExecutor(max_workers=MAX_TABLE_WORKERS) as executor:
        executor.map(process_single_table, TABLES)

def add_bucket_headers(filepath, bucket):
    '''Adds known column headers to headerless files from supported buckets.'''
    if os.path.getsize(filepath) == 0:
        return

    try:
        if bucket == 'AMF_FLAGS':
            df = pd.read_csv(filepath, header=None)
            if df.shape[1] == 4:
                df.columns = ['index', 'time', 'occurrence', 'information']
                df = df[['time', 'information']]
            elif df.shape[1] == 2:
                df.columns = ['time', 'information']
        elif bucket == 'RefPoly4':
            df = pd.read_csv(filepath, header=None)
            if df.shape[1] == 3:
                df.columns = ['time', 'channel1', 'channel2']
        else:
            return
    
        df.to_csv(filepath, index=False)
    except EmptyDataError:
        return

def get_all_file_paths(bucket, initial_remote_path, initial_local_path):
    '''Iteratively builds a flat list of all files to download using a stack.'''
    files_to_download = []
    
    # We use a list as a stack to keep track of folders we need to explore
    folders_to_explore = [(initial_remote_path, initial_local_path)]
    
    while folders_to_explore:
        current_remote, current_local = folders_to_explore.pop()
        items = list_all_items(bucket, current_remote)
        
        for item in items:
            name = item['name']
            full_path = f'{current_remote}/{name}' if current_remote else name
            local_file_path = os.path.join(current_local, name)
            
            if item['metadata'] is None:
                # It's a folder, create it and add to our stack to explore in the next loop
                os.makedirs(local_file_path, exist_ok=True)
                folders_to_explore.append((full_path, local_file_path))
            else:
                # It's a file, add to our master list
                files_to_download.append({
                    'bucket': bucket,
                    'remote_path': full_path,
                    'local_path': local_file_path
                })
                
    return files_to_download

def process_single_file(file_info):
    '''Worker function to download and process a single file.'''
    bucket = file_info['bucket']
    remote_path = file_info['remote_path']
    local_path = file_info['local_path']
    
    print(f'Downloading: {remote_path}')
    content = supabase.storage.from_(bucket).download(remote_path)
    
    with open(local_path, 'wb') as f:
        f.write(content)
        
    if bucket in ['AMF_FLAGS', 'RefPoly4']:
        add_bucket_headers(local_path, bucket)

def download_buckets():
    '''Downloads all configured storage buckets using parallel processing.'''
    print('\n--- Scanning for files (Fast phase) ---')
    all_files = []
    
    for bucket in BUCKETS:
        bucket_path = os.path.join(BUCKET_DIR, bucket)
        os.makedirs(bucket_path, exist_ok=True)
        # Gathers all files for this bucket
        all_files.extend(get_all_file_paths(bucket, '', bucket_path))

    print(f'\n--- Starting parallel download of {len(all_files)} files ---')
    # Maps the entire master list of files to the thread pool
    with ThreadPoolExecutor(max_workers=MAX_FILE_WORKERS) as executor:
        executor.map(process_single_file, all_files)

def run_sync():
    '''Refreshes the complete local data export from Supabase tables and buckets.'''
    clear_local_data()
    download_tables()
    download_buckets()
    print('\nSYNC COMPLETE')

if __name__ == '__main__':
    run_sync()