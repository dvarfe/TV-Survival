from modules.constants import TRAIN_GRID, TEST_GRID
from models.preprocessing import stratified_split, Sampler, TimeTransformer, ObservationAggregator, FeatureFilter, NanImputer, TimeMaskedScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
import pandas as pd
# Программа, которая делает предобрабатывает truncated данные и сохраняет
# train и test в csv-файлы Preprocessed/{n_samples}_{train/test}_preprocessed.csv

# Отключаю варнинги. Возможно не очень правильно так делать, но они засоряют вывод
# import warnings
# warnings.simplefilter(action='ignore')

RANDOM_STATE = 42
# RANDOM_STATE = 13
# RANDOM_STATE = 14
# RANDOM_STATE = 15

DATA_FOLDER = 'HOD_KONEM'
ID_COL = 'serial_number'
EVENT_COL = 'failure'

if __name__ == "__main__":

    df = pd.read_parquet('2016_2017_trunc.parquet')
    print('Данные прочитаны')

    # Удаляем диски с одним наблюдением, они бесполезны
    serial_numbers_to_leave = df.groupby(ID_COL).size().reset_index()
    serial_numbers_to_leave = serial_numbers_to_leave[serial_numbers_to_leave[0] > 1]
    df = df[df[ID_COL].isin(serial_numbers_to_leave[ID_COL].unique())]

    df_train, df_test = stratified_split(df, random_state=RANDOM_STATE,
                                         test_size=0.2, id_col=ID_COL, event_col=EVENT_COL)

    train_sampler = Sampler(df_train, random_state=RANDOM_STATE, mode='first', id_col=ID_COL) ### MODE!!!1
    test_sampler = Sampler(df_test, random_state=RANDOM_STATE, mode='first', id_col=ID_COL) ### MODE!!!1

    convergence_problem = []
    # Специфично для backblaze
    normalized_features = [col_name for col_name in df_train.columns if 'normalized' in col_name]
    cols_to_drop = ['model'] + convergence_problem + normalized_features
    features_to_scale = [col_name for col_name in df_train.columns if (
        'smart' in col_name) and (col_name not in cols_to_drop)] + ['capacity_bytes']

    preprocessing_pipeline = Pipeline(steps=[
        ('time', TimeTransformer(id_col=ID_COL, event_column='failure')),
        ('agg', ObservationAggregator(id_col=ID_COL)),
        ('filter', FeatureFilter(
            nan_fraction=0.8,
            features_to_remove=cols_to_drop
        )),
        ('imputer', NanImputer(id_col=ID_COL)),
        ('scaler', TimeMaskedScaler(
            features=features_to_scale,
            id_col=ID_COL
        ))
    ])

    for train_samples in TRAIN_GRID:

        df_train_n = train_sampler.get_n_samples(n_samples=train_samples)
        # df_train_n.loc[:, features_to_scale] = df_train_n[features_to_scale].astype('float')
        df_train_n = preprocessing_pipeline.fit_transform(df_train_n)
        df_train_n.to_parquet(f'{DATA_FOLDER}/{train_samples}_train_preprocessed.parquet', index=False)
        dump(preprocessing_pipeline, f'preprocessing_{train_samples}.joblib')
        print(f"train_n: {train_samples}, mean: {df_train_n.groupby('serial_number')['serial_number'].size().mean()}")
        for test_samples in TEST_GRID:
            df_test_n = test_sampler.get_n_samples(test_samples)
            # df_test_n[features_to_scale] = df_test_n[features_to_scale].astype('float')
            print(f"test_n: {test_samples}, mean: {df_test_n.groupby('serial_number')['serial_number'].size().mean()}")
            df_test_n = preprocessing_pipeline.transform(df_test_n)
            df_test_n.to_parquet(f'{DATA_FOLDER}/{train_samples}_{test_samples}_test_preprocessed.parquet', index=False)
            print(f'Вся выборка: {df.shape}, Обучающая выборка: {df_train_n.shape}, Тестовая выборка: {df_test_n.shape}')

        print(f'Обработка n_samples={train_samples} завершена')
