import os
import shutil
import pandas as pd
from scripts.supabase_client import supabase
from pandas.errors import EmptyDataError

# Configuration
DATA_DIR = "data"

TABLE_DIR = os.path.join(DATA_DIR, "tables")
BUCKET_DIR = os.path.join(DATA_DIR, "buckets")

def get_tables():

    response = supabase.rpc("get_tables").execute()

    tables = [row["table_name"] for row in response.data]

    return tables

def get_buckets():
    response = supabase.storage.list_buckets()

    buckets = [bucket.name for bucket in response]

    return buckets

'''TABLES = [
    "amf_turns",
    "backend_event_log",
    "chip_calibration_spetrca",
    "chip_calibrations",
    "device",
    "experiments",
    "flags",
    "immob",
    "machine",
    "measurements",
    "optical_reference_spectra",
    "pel_upload",
    "qc_baseline_drift",
    "qc_measurements",
    "sensorgram_history",
    "standard_curve_history",
    "standard_curves",
    "system_utilization_log",
    "uploads"
]'''

TABLES = get_tables()
BUCKETS = get_buckets()

'''BUCKETS = [
    "Chip_Calibration",
    "Optical_Reference_Spectra",
    "AMF_FLAGS",
    "RefPoly4",
    "Sensorgram",
    "Spectra"
]'''

# Clear old data
def clear_local_data():

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    os.makedirs(TABLE_DIR)
    os.makedirs(BUCKET_DIR)

    print("Old data cleared")

def list_all_items(bucket, remote_path):

    all_items = []
    offset = 0
    limit = 100

    while True:
        items = supabase.storage.from_(bucket).list(
            remote_path,
            {
                "limit": limit,
                "offset": offset,
            },
        )

        if not items:
            break

        all_items.extend(items)

        if len(items) < limit:
            break

        offset += limit

    return all_items

# Download tables
def download_tables():

    for table in TABLES:

        print(f"Downloading table: {table}")

        response = supabase.table(table).select("*").execute()

        df = pd.DataFrame(response.data)

        filepath = os.path.join(TABLE_DIR, f"{table}.csv")

        df.to_csv(filepath, index=False)

        print(f"Saved {len(df)} rows")

def add_bucket_headers(filepath, bucket):

    if os.path.getsize(filepath) == 0:
        print(f"Skipping empty file: {filepath}")
        return

    try:

        if bucket == "AMF_FLAGS":
                columns = ["time", "information"]
        
        elif bucket == "RefPoly4":
            columns = ["time", "channel1", "channel2"]
    
        else:
            return

        # Read file without headers
        df = pd.read_csv(filepath, header=None)
    
        # Add headers
        df.columns = columns
    
        # Save back
        df.to_csv(filepath, index=False)
    
        print(f"Added headers to {filepath}")
    
    except EmptyDataError:
        print(f"Warning: No columns to parse in {filepath}. Skipping.")
        return

# Download buckets
def download_buckets():

    for bucket in BUCKETS:

        print(f"\nDownloading bucket: {bucket}")

        bucket_path = os.path.join(BUCKET_DIR, bucket)

        os.makedirs(bucket_path, exist_ok=True)

        download_folder(bucket=bucket, remote_path="", local_path=bucket_path)

def download_folder(bucket, remote_path, local_path):

    # List contents of current folder
    #items = supabase.storage.from_(bucket).list(remote_path)
    items = list_all_items(bucket, remote_path)

    for item in items:

        name = item["name"]

        # Full path in Supabase
        if remote_path:
            full_path = f"{remote_path}/{name}"
        else:
            full_path = name

        # Local path
        local_file_path = os.path.join(local_path, name)

        # If metadata is None, this is a folder
        if item["metadata"] is None:

            print("Entering folder:", full_path)

            os.makedirs(local_file_path, exist_ok=True)

            download_folder(bucket, full_path, local_file_path)

        # Otherwise download the file
        else:

            print("Downloading file:", full_path)

            content = supabase.storage.from_(bucket).download(full_path)

            with open(local_file_path, "wb") as f:
                f.write(content)

            # Add missing headers for specific buckets
            if bucket in ["AMF_FLAGS", "RefPoly4"]:
                add_bucket_headers(local_file_path, bucket)

# Main
def run_sync():

    clear_local_data()

    download_tables()

    download_buckets()

    print("\nSYNC COMPLETE")

if __name__ == "__main__":
    run_sync()