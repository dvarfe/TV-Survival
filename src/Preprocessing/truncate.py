"""
Data Truncation Preprocessor

This script processes the merged Backblaze hard drive dataset to remove truncated observations. 

It performs the following operations:

1. Identifies truncated observations (drives that didn't fail by the end of observation period)
2. Removes drives with truncated observations from the dataset
3. Renames 'date' column to 'time'.
4. Identifies and removes inconsistent drives (drives that have records after failure event)  
5. Outputs cleaned dataset

Truncated observations are drives that were still operational at the end of the study period,
which can bias results if not handled properly.
"""

import pandas as pd


def identify_truncated_observations(df):
    """
    Identify drives that have truncated observations (didn't fail by end of period).
    
    Args:
        df (pd.DataFrame): Dataset with 'date', 'failure', and 'serial_number' columns
        
    Returns:
        numpy.ndarray: Array of serial numbers with truncated observations
    """
    # Find drives that didn't fail by the last observation date
    trunc_ids = df[(df['date'] == df['date'].max()) & (df['failure'] == 0)]['serial_number'].unique()
    return trunc_ids


def remove_inconsistent_drives(df):
    """
    Remove drives with data inconsistencies (drives that have records after failure event).
    
    Inconsistent drives are those that have data records after they have already failed,
    which is logically impossible since failed drives cannot continue operating.
    
    Args:
        df (pd.DataFrame): Dataset to clean
        
    Returns:
        pd.DataFrame: Cleaned dataset without inconsistent drives
    """
    # Find maximum failure status for each drive
    df['max_f'] = df.groupby('serial_number')['failure'].transform('max')
    
    # Find drives that have records after failure (inconsistent data)
    weird_disks = df[
        (df.groupby('serial_number')['time'].transform('max') == df['time']) & 
        (df['failure'] != df['max_f'])
    ]['serial_number'].unique()
    
    # Remove inconsistent drives
    df_clean = df[~df['serial_number'].isin(weird_disks)].copy()
    df_clean.drop(columns=['max_f'], inplace=True)
    
    return df_clean


def truncate_observations(input_file='merged.parquet', output_file='2016_2018_trunc.parquet'):
    """
    Main function to process and truncate observations.
    
    Args:
        input_file (str): Path to input parquet file
        output_file (str): Path to output parquet file
        
    Returns:
        pd.DataFrame: Processed dataset
    """

    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    
    initial_failure_rate = df['failure'].mean()
    print(f"Initial failure rate: {initial_failure_rate:.4f}")
    
    trunc_ids = identify_truncated_observations(df)
    print(f"Found {len(trunc_ids)} drives with truncated observations")
    
    df = df[~df['serial_number'].isin(trunc_ids)].copy()
    print(f"Removed truncated observations. Remaining drives: {df['serial_number'].nunique()}")
    
    # Rename date column to time
    df.rename(columns={'date': 'time'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Remove inconsistent drives
    initial_drives = df['serial_number'].nunique()
    df = remove_inconsistent_drives(df)
    final_drives = df['serial_number'].nunique()
    print(f"Removed {initial_drives - final_drives} inconsistent drives")
    
    # Calculate final statistics
    final_failure_rate = df['failure'].mean()
    print(f"Final failure rate: {final_failure_rate:.4f}")
    print(f"Final dataset size: {len(df)} records, {final_drives} unique drives")
    
    # Save processed dataset
    df.to_parquet(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    return df


if __name__ == "__main__":
    # Process the data
    df = truncate_observations()
    
    print(f"Dataset has {len(df)} observations")