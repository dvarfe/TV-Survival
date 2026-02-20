from survivors.metrics import ibs_remain, iauc
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np


def prepare_data_for_test(df, id_col='id', time_col='time', event_col='event', mode='independent'):
    '''
    Takes a dataset with columns id_col, time_col, event_col, features
    Returns dataframe, events, durations
    The dataframe excludes the latest observations corresponding to each drive
    '''
    durations = df.groupby(id_col)[time_col].transform('max') - df[time_col]
    events = df.groupby(id_col)[event_col].transform('max')[durations != 0]
    result_df = df.loc[durations != 0, :].drop(event_col, axis='columns')
    durations = durations[durations != 0]
    if mode == 'agg':
        last_observ = result_df[time_col] == result_df.groupby(id_col)[time_col].transform('max')
        events = events[last_observ]
        durations = durations[last_observ]
    return result_df, events.astype('bool'), durations

# Scoring functions


def concordance_by_model(model, df, times, mode='independent'):
    '''
    Takes a model and dataframe df.
    Model must have a predict method that accepts a dataframe with
    id_col, time_col, event_col, features
    This method returns a dataframe with survival function values, preserving id_col and time_col columns
    for identification
    '''
    X, events, durations = prepare_data_for_test(
        df, id_col=model.id_col, time_col=model.time_col, event_col=model.event_col, mode=mode)
    predictions = model.get_expected_time(X, times=times)
    ci = concordance_index(durations, predictions, events)
    return ci


def ibs_by_model(model, df_train, df_test, times, mode='independent'):

    # Prepare training data
    survival_train = pd.DataFrame()
    X_train, train_events, train_durations = prepare_data_for_test(
        df_train, id_col=model.id_col, time_col=model.time_col, event_col=model.event_col)
    survival_train['event'] = train_events
    survival_train['duration'] = train_durations

    # Prepare test data
    survival_test = pd.DataFrame()
    X_test, test_events, test_durations = prepare_data_for_test(
        df_test, id_col=model.id_col, time_col=model.time_col, event_col=model.event_col, mode=mode)
    survival_test['event'] = test_events
    survival_test['duration'] = test_durations

    # If times parameter is not provided explicitly, integrate over the full time interval
    if times is None:
        times = np.array(model.baseline_survival_.index)

    survival_estim = model.predict(X_test, times=times).drop([model.id_col, model.time_col], axis='columns')

    ibs = ibs_remain(
        survival_train.to_records(index=False),
        survival_test.to_records(index=False),
        survival_estim,
        times
    )
    return ibs


def get_ci_and_ibs(model, df_test, times, mode='independent'):
    '''
    mode in {independent, agg}
    Returns CI and IBS
    '''

    X_test, test_events, test_durations = prepare_data_for_test(
        df_test, id_col=model.id_col, time_col=model.time_col, event_col=model.event_col, mode=mode)
    survival_test = pd.DataFrame()
    survival_test['event'] = test_events
    survival_test['duration'] = test_durations

    X_pred = model.predict(X_test, times)
    lifetime_pred = model.get_expected_time_by_predictions(X_pred, times)

    ci = concordance_index(test_durations, lifetime_pred, test_events)

    survival_estim = X_pred.drop([model.id_col, model.time_col], axis='columns')

    ibs = ibs_remain(
        None,
        survival_test.to_records(index=False),
        survival_estim,
        times
    )
    return ci, ibs


def get_ci_and_ibs_by_data(model, X_pred, test_events, test_durations, times):

    survival_test = pd.DataFrame()
    survival_test['event'] = test_events
    survival_test['duration'] = test_durations

    lifetime_pred = model.get_expected_time_by_predictions(X_pred, times)

    ci = concordance_index(test_durations, lifetime_pred, test_events)

    survival_estim = X_pred.drop([model.id_col, model.time_col], axis='columns')
    ibs = ibs_remain(
        None,
        survival_test.to_records(index=False),
        survival_estim,
        times
    )
    return ci, ibs


def get_metrics(model, df_pred: pd.DataFrame, df_gt: pd.DataFrame, times: np.ndarray, metrics, axis=-1, df_train=None):

    metrics_dict = {}

    if df_train is not None:
        iauc_train = pd.DataFrame()
        iauc_train['cens'] = df_train['event'].astype(bool)
        iauc_train['duration'] = df_train['duration']

    survival_test = pd.DataFrame()
    survival_test['event'] = df_gt['event'].astype(bool)
    survival_test['duration'] = df_gt['duration']

    iauc_test = pd.DataFrame()
    iauc_test['cens'] = df_gt['event'].astype(bool)
    iauc_test['time'] = df_gt['duration']

    lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

    if 'ci' in metrics:
        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['event'])
        metrics_dict['ci'] = ci

    survival_estim = df_pred.drop(['id', 'time'], axis='columns')

    if 'ibs' in metrics:
        # survival_test.to_csv('test_new.csv')
        # survival_estim.to_csv('estim_new.csv')
        # raise ValueError()

        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times,
            axis=axis
        )
        metrics_dict['ibs'] = ibs

    if 'iauc' in metrics:
        if df_train is None:
            raise ValueError("df_train must be provided to compute iauc")
        hazard_estim = -np.log(survival_estim)
        iauc_score = iauc(
            iauc_train.to_records(index=False),
            iauc_test.to_records(index=False),
            hazard_estim,
            times,
        )
        metrics_dict['iauc'] = iauc_score

    return metrics_dict
