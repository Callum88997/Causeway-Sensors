# Imports required packages
import os
import shutil
import pandas as pd
from scripts.supabase_client import supabase
from pandas.errors import EmptyDataError

# Defines configuration constants for data directories
DATA_DIR = "data"

TABLE_DIR = os.path.join(DATA_DIR, "tables")
BUCKET_DIR = os.path.join(DATA_DIR, "buckets")

def get_tables():
    '''Retrieves the names of all accessible Supabase database tables.

    Returns:
        list[str]: Available table names.
    '''

    # Executes the RPC call to retrieve table names
    response = supabase.rpc("get_tables").execute()

    # Extracts table names from the response data
    tables = [row["table_name"] for row in response.data]

    # Returns the list of tables
    return tables

def get_buckets():
    '''Retrieves the names of all accessible Supabase storage buckets.

    Returns:
        list[str]: Available storage-bucket names.
    '''

    # Fetches the list of buckets from the Supabase storage
    response = supabase.storage.list_buckets()

    # Extracts bucket names from the response
    buckets = [bucket.name for bucket in response]

    # Returns the list of buckets
    return buckets

# Retrieves and stores the available tables and buckets
TABLES = get_tables()
BUCKETS = get_buckets()

def clear_local_data():
    '''Removes the previous local export and recreates the required directories.'''

    # Removes the existing data directory if it exists
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    # Recreates the necessary table and bucket directories
    os.makedirs(TABLE_DIR)
    os.makedirs(BUCKET_DIR)

    # Prints a confirmation message
    print("Old data cleared")

def list_all_items(bucket, remote_path):
    '''Retrieves every item in a storage path, including paginated results.

    Args:
        bucket (str): Name of the Supabase storage bucket.
        remote_path (str): Path within the bucket to enumerate.

    Returns:
        list[dict]: Metadata for every item found at the requested path.
    '''

    # Initialises variables for pagination
    all_items = []
    offset = 0
    limit = 100

    # Continues processing until all available items have been handled
    while True:
        
        # Fetches a batch of items from the specified bucket and path
        items = supabase.storage.from_(bucket).list(remote_path, {"limit": limit, "offset": offset,})

        # Breaks the loop if no items are returned
        if not items:
            break

        # Appends the retrieved items to the complete list
        all_items.extend(items)

        # Breaks the loop if the number of retrieved items is less than the limit
        if len(items) < limit:
            break

        # Increments the offset for the next pagination batch
        offset += limit

    # Returns the complete list of items
    return all_items

def download_tables():
    '''Downloads every configured database table and saves it as a local CSV file.'''

    # Loops through each available table
    for table in TABLES:

        # Prints the current table being downloaded
        print(f"Downloading table: {table}")

        # Executes a select query to retrieve all records from the table
        response = supabase.table(table).select("*").execute()

        # Converts the response data into a pandas dataframe
        df = pd.DataFrame(response.data)

        # Defines the local filepath for the CSV output
        filepath = os.path.join(TABLE_DIR, f"{table}.csv")

        # Saves the dataframe to a CSV file without the index
        df.to_csv(filepath, index=False)

        # Prints the number of rows saved
        print(f"Saved {len(df)} rows")

def add_bucket_headers(filepath, bucket):
    '''Adds known column headers to headerless files from supported buckets.

    Args:
        filepath (str): Local path to the downloaded file.
        bucket (str): Source bucket used to select the expected schema.
    '''

    # Checks if the file is empty and skips processing if true
    if os.path.getsize(filepath) == 0:
        
        # Prints a skipping message
        print(f"Skipping empty file: {filepath}")
        
        # Returns control to the calling code
        return

    # Attempts to parse and add headers based on the bucket name
    try:

        # Assigns column headers for the AMF_FLAGS bucket
        if bucket == "AMF_FLAGS":
                columns = ["time", "information"]
        
        # Assigns column headers for the RefPoly4 bucket
        elif bucket == "RefPoly4":
            columns = ["time", "channel1", "channel2"]
    
        # Returns control to the calling code for unsupported buckets
        else:
            return

        # Reads the headerless CSV file into a dataframe
        df = pd.read_csv(filepath, header=None)
    
        # Applies the assigned column headers to the dataframe
        df.columns = columns
    
        # Saves the updated dataframe back to the CSV file
        df.to_csv(filepath, index=False)
    
        # Prints a confirmation message
        print(f"Added headers to {filepath}")
    
    # Catches empty data errors and skips the file
    except EmptyDataError:
        
        # Prints a warning message
        print(f"Warning: No columns to parse in {filepath}. Skipping.")
        
        # Returns control to the calling code
        return

def download_buckets():
    '''Downloads all configured storage buckets into the local export directory.'''

    # Loops through each available storage bucket
    for bucket in BUCKETS:

        # Prints the current bucket being downloaded
        print(f"\nDownloading bucket: {bucket}")

        # Defines the local directory path for the bucket
        bucket_path = os.path.join(BUCKET_DIR, bucket)

        # Creates the bucket directory if it does not already exist
        os.makedirs(bucket_path, exist_ok=True)

        # Initiates the recursive folder download for the bucket
        download_folder(bucket=bucket, remote_path="", local_path=bucket_path)

def download_folder(bucket, remote_path, local_path):
    '''Recursively downloads a remote storage folder and preserves its hierarchy.

    Args:
        bucket (str): Name of the source storage bucket.
        remote_path (str): Current path within the remote bucket.
        local_path (str): Matching local directory where files are written.
    '''

    # Retrieves all items within the current remote path
    items = list_all_items(bucket, remote_path)

    # Loops through each retrieved item
    for item in items:

        # Extracts the name of the item
        name = item["name"]

        # Constructs the full remote path for the item
        if remote_path:
            full_path = f"{remote_path}/{name}"
        else:
            full_path = name

        # Constructs the corresponding local file path
        local_file_path = os.path.join(local_path, name)

        # Checks if the item is a folder (metadata is None)
        if item["metadata"] is None:

            # Prints the folder being entered
            print("Entering folder:", full_path)

            # Creates the local directory for the folder
            os.makedirs(local_file_path, exist_ok=True)

            # Recursively calls the function to download the folder contents
            download_folder(bucket, full_path, local_file_path)

        # Processes the item as a file if metadata exists
        else:

            # Prints the file being downloaded
            print("Downloading file:", full_path)

            # Downloads the file content from the Supabase bucket
            content = supabase.storage.from_(bucket).download(full_path)

            # Opens the local file safely and writes the downloaded content
            with open(local_file_path, "wb") as f:
                f.write(content)

            # Adds missing headers if the file belongs to specific buckets
            if bucket in ["AMF_FLAGS", "RefPoly4"]:
                add_bucket_headers(local_file_path, bucket)

def run_sync():
    '''Refreshes the complete local data export from Supabase tables and buckets.'''

    # Clears any existing local data
    clear_local_data()

    # Downloads all tables from the database
    download_tables()

    # Downloads all files from the storage buckets
    download_buckets()

    # Prints a completion message
    print("\nSYNC COMPLETE")

# Executes the synchronisation process if the script is run directly
if __name__ == "__main__":
    run_sync()
    