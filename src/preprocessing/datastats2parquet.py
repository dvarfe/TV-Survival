"""
Data Statistics to Parquet Converter

This script processes Backblaze hard drive failure datasets downloaded from their official website:
https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data

The script converts CSV files to Parquet format for improved performance and storage efficiency.
It performs the following operations:

1. Unzips compressed Backblaze data archives containing CSV files
2. Converts CSV files to Parquet format with proper data type casting:
   - Converts date columns to datetime format
   - Converts failure column to boolean
   - Converts capacity_bytes to int64
   - Converts SMART attributes (smart_*) to nullable integers
3. Aggregates data from multiple years by:
   - Randomly selecting a subset of unique serial numbers (default: 5000)
   - Filtering records to include failure events and periodic samples
   - Merging data across multiple parquet files
4. Creates a consolidated dataset

The script is designed for preprocessing large-scale hard drive reliability datasets
typically used in survival analysis and failure prediction studies.
"""

import os
import zipfile
import pandas as pd
import numpy as np
from shutil import rmtree


def folder_to_parquet(folder_path, save_to):
    """
    Convert all CSV files in a folder to a single Parquet file.
    
    Args:
        folder_path (str): Path to folder containing CSV files
    """
    df_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            df['date'] = pd.to_datetime(df.date)
            df['failure'] = df['failure'].astype('bool')
            df['capacity_bytes'] = df['capacity_bytes'].astype('int64')
            for col in [smart_col for smart_col in df.columns if smart_col.startswith('smart_')]:
                df[col] = df[col].astype('Int64')
            df_list.append(df)
    df_concat = pd.concat(df_list)
    os.makedirs(save_to, exist_ok=True)
    df_concat.to_parquet(os.path.join(save_to, f'{os.path.basename(folder_path)}.parquet'))


def unzip_folder(zip_file, extraction_folder_path):
    """
    Unzip a file and remove the original zip file.
    
    Args:
        zip_file (str): Path to zip file
        extraction_folder_path (str): Path where to extract files
    """
    if zip_file.endswith('.zip'):
        year = os.path.basename(zip_file).split('.')[0][::-1].split('_', 1)[0][::-1]
        extract_to = extraction_folder_path
        if int(year) > 2015:
            extract_to = os.path.join(extraction_folder_path, os.path.basename(zip_file).split('.')[0])
        print(f'Unzipping {zip_file}')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        # os.remove(zip_file)
        return extract_to
    return None


def convert_to_parquet(folder_path, extract_path, save_parquet):
    """
    Process all zip files in a folder: unzip, convert to parquet, and cleanup.
    
    Args:
        folder_path (str): Path to folder containing zip files
    """
    for zipfile in os.listdir(folder_path):
        if zipfile.endswith('.zip'):
            zip_path = os.path.join(folder_path, zipfile)
            # unzipped_path = os.path.join(extract_path, zipfile.split('.')[0])
            unzipped_path = unzip_folder(zip_path, extract_path)
            rmtree(os.path.join(unzipped_path, '__MACOSX'))
            print(f'Converting to parquet {unzipped_path}')
            folder_to_parquet(unzipped_path, save_parquet)
            print(f'Removing {unzipped_path}')
            rmtree(unzipped_path)


def agg_data(data_folder='.', unique_ids=5000, frequency=1, sample_file='2016.parquet'):
    """
    Aggregate data from multiple parquet files with sampling and filtering.
    
    Data Aggregation Logic:
    1. Randomly selects n drives from the first dataset (sample_file)
    2. Tracks the complete behavioral history of these selected drives across all years
    3. For each selected drive, includes:
       - All failure events (failure=1)  
       - Periodic samples (every N days based on frequency parameter)
    
    This approach ensures we have longitudinal data for a manageable subset of drives
    while preserving both failure events and regular monitoring data.
    
    Args:
        data_folder (str): Path to folder containing parquet files
        unique_ids (int): Number of unique serial numbers to sample
        frequency (int): Frequency for sampling non-failure records (every N days)
        sample_file (str): Parquet file to use for sampling serial numbers 
                          (The first file chronologically)    
    Returns:
        pd.DataFrame: Aggregated dataset with filtered records    
    """
    # Step 1: Select random sample of drives from the first dataset
    df = pd.read_parquet(os.path.join(data_folder, sample_file))
    serial_numbers = np.random.choice(df['serial_number'].unique(), unique_ids)
    
    # Step 2: Track behavioral history of selected drives across all datasets
    df_list = []
    for file in os.listdir(data_folder):
        if file.endswith('.parquet'):
            print(f'Processing {file}')
            df = pd.read_parquet(os.path.join(data_folder, file))
            # Filter to only selected drives
            df = df[df['serial_number'].isin(serial_numbers)]
            # Keep failure events + periodic samples for monitoring
            df = df[(df['failure'] == 1) | (df['date'].dt.day % frequency == 0)]
            df_list.append(df)
    df = pd.concat(df_list)
    return df

def main(data_folder='.', unique_ids=5000, frequency=1, sample_file='2016.parquet', extract_folder='tmp', parquet_folder='Parquet'):
    # Step 1: Convert zip archives to parquet format
    print("Converting zip archives to parquet format...")
    convert_to_parquet(data_folder, extract_folder, parquet_folder)
    
    # Step 2: Aggregate data from parquet files
    print("Aggregating data from parquet files...")
    df = agg_data(parquet_folder, unique_ids, frequency, sample_file)
    
    # Step 3: Print statistics
    print(f"Number of unique serial numbers: {df['serial_number'].unique().shape[0]}")
    
    # Get last records for each serial number
    last = df[df['date'] == df.groupby('serial_number')['date'].transform('max')]
    print(f"Number of last records: {len(last)}")
    
    # Save merged data to parquet
    df.to_parquet(os.path.join(data_folder, 'merged.parquet'))
    print("Merged data saved to merged.parquet")

if __name__ == "__main__":
    main()