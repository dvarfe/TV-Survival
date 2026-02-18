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


def folder_to_parquet(folder_path):
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
    df_concat.to_parquet(f'{os.path.basename(folder_path)}.parquet')


def unzip_folder(zip_file, extraction_folder_path):
    """
    Unzip a file and remove the original zip file.
    
    Args:
        zip_file (str): Path to zip file
        extraction_folder_path (str): Path where to extract files
    """
    if zip_file.endswith('.zip'):
        print(f'Unzipping {zip_file}')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extraction_folder_path)
        os.remove(zip_file)


def convert_to_parquet(folder_path):
    """
    Process all zip files in a folder: unzip, convert to parquet, and cleanup.
    
    Args:
        folder_path (str): Path to folder containing zip files
    """
    for zipfile in os.listdir(folder_path):
        if zipfile.endswith('.zip'):
            unzipped_path = zipfile.split('.')[0]
            unzip_folder(zipfile, unzipped_path)
            rmtree(os.path.join(unzipped_path, '__MACOSX'))
            print(f'Converting to parquet {unzipped_path}')
            folder_to_parquet(unzipped_path)
            print(f'Removing {unzipped_path}')
            rmtree(unzipped_path)


def agg_data(data_folder='.', unique_ids=5000, frequency=1, sample_file='2016.parquet'):
    """
    Aggregate data from multiple parquet files with sampling and filtering.
    
    Args:
        data_folder (str): Path to folder containing parquet files
        unique_ids (int): Number of unique serial numbers to sample
        frequency (int): Frequency for sampling non-failure records (every N days)
        sample_file (str): Parquet file to use for sampling serial numbers 
                          (The first file chronologically)    
    Returns:
        pd.DataFrame: Aggregated dataset with filtered records    
    """
    df = pd.read_parquet(sample_file)
    serial_numbers = np.random.choice(df['serial_number'].unique(), unique_ids)
    df_list = []
    for file in os.listdir(data_folder):
        if file.endswith('.parquet'):
            print(f'Processing {file}')
            df = pd.read_parquet(file)
            df = df[df['serial_number'].isin(serial_numbers)]
            df = df[(df['failure'] == 1) | (df['date'].dt.day % frequency == 0)]
            df_list.append(df)
    df = pd.concat(df_list)
    return df


if __name__ == "__main__":
    # Step 1: Convert zip archives to parquet format
    print("Converting zip archives to parquet format...")
    convert_to_parquet('.')
    
    # Step 2: Aggregate data from parquet files
    print("Aggregating data from parquet files...")
    df = agg_data()
    
    # Step 3: Print statistics
    print(f"Number of unique serial numbers: {df['serial_number'].unique().shape[0]}")
    
    # Get last records for each serial number
    last = df[df['date'] == df.groupby('serial_number')['date'].transform('max')]
    print(f"Number of last records: {len(last)}")
    
    # Save merged data to parquet
    df.to_parquet('merged.parquet')
    print("Merged data saved to merged.parquet")