from modules.constants import TRAIN_GRID, TEST_GRID
from modules.preprocessing import stratified_split, Sampler, TimeTransformer, ObservationAggregator, FeatureFilter, NanImputer, TimeMaskedScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Program that preprocesses truncated data and saves train/test sets to parquet files
# Format: {DATA_FOLDER}/{n_samples}_{train/test}_preprocessed.parquet


def main(input_file='2016_2017_trunc.parquet', 
         data_folder='Preprocessed',
         random_state=42,
         test_size=0.2,
         id_col='serial_number',
         event_col='failure',
         mode='random',
         nan_fraction=0.8):
    """
    Main function to preprocess and split dataset.
    
    Args:
        input_file (str): Input truncated parquet file
        data_folder (str): Output directory for processed files  
        random_state (int): Random state for reproducibility
        test_size (float): Test set proportion
        id_col (str): ID column name
        event_col (str): Event column name
        mode (str): Sampling mode for Sampler
        nan_fraction (float): NaN fraction threshold for FeatureFilter
    """

    df = pd.read_parquet(input_file)
    print('Data loaded')

    # Remove drives with single observation - they are not useful
    serial_numbers_to_leave = df.groupby(id_col).size().reset_index()
    serial_numbers_to_leave = serial_numbers_to_leave[serial_numbers_to_leave[0] > 1]
    df = df[df[id_col].isin(serial_numbers_to_leave[id_col].unique())]

    df_train, df_test = stratified_split(df, random_state=random_state,
                                         test_size=test_size, id_col=id_col, event_col=event_col)

    train_sampler = Sampler(df_train, random_state=random_state, mode=mode, id_col=id_col)
    test_sampler = Sampler(df_test, random_state=random_state, mode=mode, id_col=id_col)

    # Backblaze-specific feature filtering
    normalized_features = [col_name for col_name in df_train.columns if 'normalized' in col_name]
    cols_to_drop = ['model'] + normalized_features
    features_to_scale = [col_name for col_name in df_train.columns if (
        'smart' in col_name) and (col_name not in cols_to_drop)] + ['capacity_bytes']

    preprocessing_pipeline = Pipeline(steps=[
        ('time', TimeTransformer(id_col=id_col, event_column=event_col)),
        ('agg', ObservationAggregator(id_col=id_col)),
        ('filter', FeatureFilter(
            nan_fraction=nan_fraction,
            features_to_remove=cols_to_drop
        )),
        ('imputer', NanImputer(id_col=id_col)),
        ('scaler', TimeMaskedScaler(
            features=features_to_scale,
            id_col=id_col
        ))
    ])

    for train_samples in TRAIN_GRID:

        df_train_n = train_sampler.get_n_samples(n_samples=train_samples)
        df_train_n = preprocessing_pipeline.fit_transform(df_train_n)
        df_train_n.to_parquet(f'{data_folder}/{train_samples}_train_preprocessed.parquet', index=False)
        print(f"train_n: {train_samples}, mean: {df_train_n.groupby(id_col)[id_col].size().mean()}")
        
        for test_samples in TEST_GRID:
            df_test_n = test_sampler.get_n_samples(test_samples)
            print(f"test_n: {test_samples}, mean: {df_test_n.groupby(id_col)[id_col].size().mean()}")
            df_test_n = preprocessing_pipeline.transform(df_test_n)
            df_test_n.to_parquet(f'{data_folder}/{train_samples}_{test_samples}_test_preprocessed.parquet', index=False)
            print(f'Total dataset: {df.shape}, Training set: {df_train_n.shape}, Test set: {df_test_n.shape}')

        print(f'Processing n_samples={train_samples} completed')


if __name__ == "__main__":
    main()
