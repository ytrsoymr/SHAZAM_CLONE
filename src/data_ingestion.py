import sqlite3
import pandas as pd
import zipfile
import io
import re
import os

# Connect to the SQLite database
conn = sqlite3.connect(r'data\eng_subtitles_database.db')

# Read data from the table (including num for reference)
df = pd.read_sql_query("SELECT num, content FROM zipfiles LIMIT 100", conn)

# Define a function to extract content from binary ZIP files
def decode_method(binary_data):
    """Extracts and decodes subtitle content from a compressed binary file."""
    try:
        with io.BytesIO(binary_data) as f:
            with zipfile.ZipFile(f, 'r') as zip_file:
                # Extract the first file from the archive
                file_name = zip_file.namelist()[0]  # Assumes at least one file exists
                subtitle_content = zip_file.read(file_name)
                return subtitle_content.decode('utf-8', errors='replace')  # Safe decoding
    except Exception as e:
        print(f"Error decoding ZIP file: {e}")
        return None  # Return None for failed extractions

# Apply the function to extract subtitles
df['file_content'] = df['content'].apply(decode_method)

# Define a function to clean subtitle text
def clean_text(text):
    """Cleans subtitle text by removing timestamps and unwanted characters."""
    if pd.isna(text):
        return ""
    
    # Remove subtitle timestamps (e.g., "00:02:15,000 --> 00:02:18,000")
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
    
    # Remove any non-alphanumeric characters except basic punctuation
    text = re.sub(r'[^a-zA-Z0-9 .,!?]', '', text)
    
    return text.strip()

# Apply text cleaning
df["cleaned_text"] = df["file_content"].apply(clean_text)

# Save processed subtitles with `num` column for reference
output_path = "data/cleaned_subtitles.csv"
df[["num", "cleaned_text"]].to_csv(output_path, index=False)

print(f"Processed subtitle data saved to {output_path}")
