"""Model training and evaluation script with parameter aggregation."""

import copy
import os
import pickle
import shutil
import time
from typing import List, Dict

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from scoring import ModelScorer
from utils.Dataset import DiskDataset
from .Experiments import TIMES, TRAIN_BATCHSIZE
from aggregation.PredictionsAggregator import PredictionsAggregator

np.random.seed(42)


def create_schema(base_schema, metrics_list, model_name):
    """Create evaluation schema with metrics columns."""
    ret_schema = copy.deepcopy(base_schema)
    for metric in metrics_list:
        ret_schema[f'{metric}_test'] = []
    return ret_schema


def create_res_file(filename):
    """Create results CSV file with schema."""
    schema = SCHEMA
    df = pd.DataFrame(schema)
    df.to_csv(filename, index=False)


def load_model(filename):
    """Load model from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_dict(filename, dict_to_save):
    """Write dictionary to CSV file."""
    df = pd.DataFrame(dict_to_save)
    df.to_csv(filename, mode='a', header=False, index=False)


def create_description_file():
    """Create experiment description file."""
    desc_path = os.path.join(RES_FOLDER, "Description.txt")
    with open(desc_path, "w") as f:
        f.write(EXP_DESC)


def sample_first_observations(df, sample_grid):
    """Sample first N observations for each series according to sample grid."""
    dict_of_all_samples = {}
    df = df.sort_values(by=['serial_number', 'time'])
    
    for first_n in sample_grid:
        df_train_sampled = (
            df.groupby('serial_number')
            .head(max(sample_grid))
            .groupby('serial_number')
            .tail(first_n)
        )
        df_train_sampled = df_train_sampled.drop_duplicates().sort_values(
            by=['serial_number', 'time'])
        dict_of_all_samples[first_n] = df_train_sampled
        
    return dict_of_all_samples


def get_models_from_exp(base_exp_num: int) -> List[str]:
    """Get model file paths from base experiment."""
    base_exp_folder = os.path.join("Artifacts", f'Exp_{base_exp_num}', 'models')
    return [
        os.path.join(base_exp_folder, m) 
        for m in os.listdir(base_exp_folder) 
        if m.endswith(".pkl")
    ]


def copy_models(models_path: List[str], dest_folder: str):
    """Copy model files to destination folder."""
    for model in models_path:
        if not os.path.exists(model):
            raise ValueError(f"Model doesn't exist: {model}")
        shutil.copy(model, dest_folder)


def init_experiments_folder(base_exp_num):
    """Initialize experiment folder structure and copy required files."""
    if os.path.exists(RES_FILENAME):
        raise ValueError('Results file already exists!')
        
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)
        
    models_path = get_models_from_exp(base_exp_num)
    copy_models(models_path, MODELS_FOLDER)
    create_res_file(RES_FILENAME)
    create_description_file()
    shutil.copy(f"Artifacts/Exp_{BASE_EXP_NUM}/grid_search.csv", RES_FOLDER)


def get_X_gt_long_format(df):
    """Prepare ground truth data for evaluation."""
    df = df.sort_values(['serial_number', 'time'])
    X_gt = df.copy()
    X_gt['event_time'] = X_gt['max_lifetime']
    X_gt['duration'] = X_gt['event_time'] - X_gt['time']
    X_gt = X_gt.loc[
        X_gt['duration'] != 0, 
        ['time', 'serial_number', 'duration', 'failure']
    ]
    return X_gt


def get_extended_times(df, agg_dict, times):
    """Get extended time grid from any aggregator."""
    any_method = list(agg_dict.keys())[0]
    any_weight = list(agg_dict[any_method].keys())[0]
    times_extended = agg_dict[any_method][any_weight].get_extended_times(df, times)
    return times_extended


def calculate_metrics(
    model, predictions_aggregator, scorer, method, weight, n_samples, 
    timeshift, X, X_gt, model_name, model_train_samples, times, 
    metric_postfix='test', metrics_list=['ci', 'ibs'], df_train_gt=None
):
    """Calculate metrics for given aggregation method and parameters."""
    aggregated_pred = predictions_aggregator.predict(X, times, timeshift=timeshift)

    metrics = scorer.get_metrics(
        model, aggregated_pred, X_gt, times,
        metrics=metrics_list,
        df_train=df_train_gt)

    result_row = {
        'train_samples': model_train_samples,
        'agg_samples': n_samples,
        'method': model_name.split('_')[1],
        'agg_method': method,
        'agg_weight': weight,
        'model_id': model_name
    }

    for metric in metrics:
        result_row[f'{metric}_{metric_postfix}'] = metrics.get(metric)

    return result_row


def eval_model(
    data_path, data_folder, models_folder, model_name, agg_dict, 
    metrics_list, sample_grid, times, model_train_samples, 
    metric_postfix='test', data_ext='.csv', train_batchsize=512
):
    """
    Evaluate model on data with different aggregation methods.

    Args:
        data_path: Path to evaluation data
        data_folder: Folder containing data files  
        models_folder: Folder containing model files
        model_name: Name of model to load
        agg_dict: Dictionary of aggregators
        metrics_list: List of metrics to compute
        sample_grid: Grid of sample sizes
        times: Time points for evaluation
        model_train_samples: Number of training samples
        metric_postfix: Postfix for metric names
        data_ext: Data file extension
        train_batchsize: Batch size for data loading

    Returns:
        pd.DataFrame: DataFrame with metrics, aggregation methods and weights
    """

    # Load data
    if data_ext == '.csv':
        df_data = pd.read_csv(data_path)
    else:
        df_data = pd.read_parquet(data_path)

    # Remove last observations in each series
    df_data = df_data[df_data['time'] != df_data['max_lifetime']]
    df_data = df_data.sort_values(by=['serial_number', 'time'])

    dl_data = DataLoader(
        dataset=DiskDataset('score', [data_path]),
        batch_size=train_batchsize)

    times_extended = get_extended_times(df_data, agg_dict, times)

    # Load training data if IAUC metric is needed
    if 'iauc' in metrics_list:
        train_path = os.path.join(data_folder, f'1_train_preprocessed{data_ext}')
        if data_ext == '.csv':
            df_train = pd.read_csv(train_path)
        else:
            df_train = pd.read_parquet(train_path)
        df_train_gt = get_X_gt_long_format(df_train)
    else:
        df_train_gt = None

    # Load model
    model_path = os.path.join(models_folder, model_name)
    if os.path.exists(f'{model_path}_model.pkl'):
        model = load_model(f'{model_path}_model.pkl')
    else:
        model = load_model(f'{model_path}.pkl')

    if model_name.endswith('DDH'):

        scorer = ModelScorer()
        results_list = []

        for n_samples in SAMPLE_GRID:
            print(f'Processing {n_samples} samples...')
            time_start = time.time()

            X_pred, X_gt = model.predict(
                dl_data, times=times, 
                agg_horizon=n_samples, 
                trunc_right=max(sample_grid)
            )

            metrics = scorer.get_metrics(
                model, X_pred, X_gt, times,
                metrics=metrics_list,
                df_train=df_train_gt)

            result_row = {
                'train_samples': model_train_samples,
                'agg_samples': n_samples,
                'method': model_name.split('_')[1],
                'agg_method': 'DDH',
                'agg_weight': -1,
                'model_id': model_name
            }

            for metric in metrics:
                result_row[f'{metric}_{metric_postfix}'] = metrics.get(metric)
            results_list.append(result_row)

    else:
        X_pred, X_gt = model.predict(dl_data, times=times_extended)
        X_pred = X_pred.sort_values(by=['serial_number', 'time'])
        X_gt = X_gt.sort_values(by=['serial_number', 'time'])

        sampled_predictions = sample_first_observations(X_pred, sample_grid)
        scorer = ModelScorer()
        results_list = []

        for n_samples in sample_grid:
            print(f'Processing {n_samples} samples...')
            time_start = time.time()

            cur_X_pred = sampled_predictions[n_samples]
            cur_timeshift = (
                cur_X_pred.groupby('serial_number')['time'].transform('max') 
                - cur_X_pred['time']
            )

            # Keep only last observation for metrics
            cur_X_gt = X_gt.loc[cur_X_pred.index, :]
            cur_X_gt = cur_X_gt[
                cur_X_gt['time'] == 
                cur_X_gt.groupby('serial_number')['time'].transform('max')
            ]

            for method in agg_dict:
                for weight in agg_dict[method]:
                    result_row = calculate_metrics(
                        model, agg_dict[method][weight], scorer, method, weight, 
                        n_samples, cur_timeshift, cur_X_pred, cur_X_gt, model_name, 
                        model_train_samples, times, metric_postfix, metrics_list, 
                        df_train_gt
                    )
                    results_list.append(result_row)

            print(f'Processing {n_samples} completed in {time.time() - time_start:.1f} seconds')

    return pd.DataFrame(results_list)


if __name__ == "__main__":
    """
    Main execution script for model evaluation with prediction aggregation.
    
    The PredictionsAggregator now aggregates pre-computed predictions.
    We first get the extended time scale, then build predictions using the model.
    Predictions are built once, then sliced into different lengths 
    (from 8 to 10, from 7 to 10... from 1 to 10 observations)
    using sample_first_observation, then aggregate.
    """
    BASE_SCHEMA: Dict[str, list] = {
        'train_samples': [],
        'agg_samples': [],
        'method': [],
        'agg_method': [],
        'agg_weight': [],
        'model_id': []
    }

    # Configuration
    METRICS_LIST = {'ci', 'ibs'}
    DATASET_TRAIN_SAMPLES = 20
    MODEL_TRAIN_SAMPLES = 20
    SAMPLE_GRID = [1, 10]
    EXP_NUM = 106
    BASE_EXP_NUM = 105  # Experiment number from which fitted models are taken
    DATA_FOLDER = os.path.join("Data", "Preprocessed_new")
    RES_FOLDER = os.path.join("Artifacts", f"Exp_{EXP_NUM}")
    MODELS_FOLDER = os.path.join(RES_FOLDER, "models")
    RES_FILENAME = os.path.join(
        RES_FOLDER, 
        f"Agg_{DATASET_TRAIN_SAMPLES}_{MODEL_TRAIN_SAMPLES}_{max(SAMPLE_GRID)}.csv"
    )
    TEST_SAMPLES = [25]
    MODELS_SIZE = [2048]
    MODEL_NAME = 'DDH'
    SCHEMA = create_schema(BASE_SCHEMA, METRICS_LIST, MODEL_NAME)
    DATA_EXT = '.csv'
    EXP_DESC = f"Aggregation {MODEL_NAME}"

    AGGREGATORS_DICT = {
        "n_dist": {
            "0.01": PredictionsAggregator(mode='n_dist', weight=0.01),
            "0.1": PredictionsAggregator(mode='n_dist', weight=0.1),
            "0.3": PredictionsAggregator(mode='n_dist', weight=0.3),
            "0.5": PredictionsAggregator(mode='n_dist', weight=0.5),
            "0.7": PredictionsAggregator(mode='n_dist', weight=0.7),
            "0.9": PredictionsAggregator(mode='n_dist', weight=0.9),
            "0.99": PredictionsAggregator(mode='n_dist', weight=0.99)
        },
        "t_dist": {
            "0.1": PredictionsAggregator(mode='t_dist', weight=0.1),
            "1": PredictionsAggregator(mode='t_dist', weight=1),
            "10": PredictionsAggregator(mode='t_dist', weight=10),
            "25": PredictionsAggregator(mode='t_dist', weight=25),
            "50": PredictionsAggregator(mode='t_dist', weight=50),
            "100": PredictionsAggregator(mode='t_dist', weight=100),
            "1000": PredictionsAggregator(mode='t_dist', weight=1000)
        },
        "prob_dist": {
            "-1": PredictionsAggregator(mode='prob_dist'),
        },
        "geom": {
            "0.01": PredictionsAggregator(mode='geom', weight=0.01),
            "0.1": PredictionsAggregator(mode='geom', weight=0.1),
            "0.3": PredictionsAggregator(mode='geom', weight=0.3),
            "0.5": PredictionsAggregator(mode='geom', weight=0.5),
            "0.7": PredictionsAggregator(mode='geom', weight=0.7),
            "0.9": PredictionsAggregator(mode='geom', weight=0.9),
            "0.99": PredictionsAggregator(mode='geom', weight=0.99)
        },
    }
    init_experiments_folder(BASE_EXP_NUM)
    model_name = '0_DDH'

    train_path = os.path.join(DATA_FOLDER, f'{DATASET_TRAIN_SAMPLES}_train_preprocessed{DATA_EXT}')

    train_res = eval_model(
        data_path=train_path,
        data_folder=DATA_FOLDER,
        models_folder=MODELS_FOLDER,
        model_name=model_name,
        agg_dict=AGGREGATORS_DICT,
        metrics_list=METRICS_LIST,
        sample_grid=SAMPLE_GRID,
        times=TIMES,
        model_train_samples=MODEL_TRAIN_SAMPLES,
        metric_postfix='train',
        data_ext=DATA_EXT
    )

    test_path = os.path.join(DATA_FOLDER, f'{DATASET_TRAIN_SAMPLES}_{TEST_SAMPLES[0]}_test_preprocessed{DATA_EXT}')

    test_res = eval_model(
        data_path=train_path,
        data_folder=DATA_FOLDER,
        models_folder=MODELS_FOLDER,
        model_name=model_name,
        agg_dict=AGGREGATORS_DICT,
        metrics_list=METRICS_LIST,
        sample_grid=SAMPLE_GRID,
        times=TIMES,
        model_train_samples=MODEL_TRAIN_SAMPLES,
        metric_postfix='test',
        data_ext=DATA_EXT
    )

    results_df = pd.merge(train_res, test_res, how='outer', on=[
        'agg_samples', 'train_samples', 'method', 'agg_method', 'agg_weight', 'model_id'])

    # Save results
    results_df.to_csv(RES_FILENAME, index=False)
    print(f"Results saved to {RES_FILENAME}")
