# Imports
from supabase import create_client
from dotenv import load_dotenv
import os

# Load variables from the local environment file
load_dotenv()

# Read the project URL and access key from the environment
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Create the shared Supabase client used by the synchronisation scripts
supabase = create_client(url, key)
