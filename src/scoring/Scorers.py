from typing import Tuple, Set

import pandas as pd
import numpy as np
from lifelines.utils import concordance_index  # type: ignore
from survivors.metrics import ibs_remain, iauc_WW_TI


class ModelScorer():

    def get_ci_and_ibs(self, model, df_pred: pd.DataFrame, df_gt: pd.DataFrame, times: np.ndarray) -> Tuple[float, float]:
        """Calculate Concordance Index (CI) and Integrated Brier Score (IBS).

        Args:
            model: The trained model used for predictions.
            df_pred (pd.DataFrame): DataFrame containing predicted survival functions.
            df_gt (pd.DataFrame): DataFrame containing ground truth durations and event indicators.
            times (np.ndarray): Array of time points for evaluation.

        Returns:
            Tuple[float, float]: Concordance Index (CI) and Integrated Brier Score (IBS).
        """
        survival_test = pd.DataFrame()
        survival_test['event'] = df_gt['failure'].astype(bool)
        survival_test['duration'] = df_gt['duration']

        lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])

        survival_estim = df_pred.drop(['serial_number', 'time'], axis='columns')
        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times
        )
        return ci, ibs

    def get_ci_and_ibs_agg(self, model, df_pred: pd.DataFrame, df_gt: pd.DataFrame, times: np.ndarray) -> Tuple[float, float]:
        """Calculate Concordance Index (CI) and Integrated Brier Score (IBS).

        Args:
            model: The trained model used for predictions.
            df_pred (pd.DataFrame): DataFrame containing predicted survival functions.
            df_gt (pd.DataFrame): DataFrame containing ground truth durations and event indicators.
            times (np.ndarray): Array of time points for evaluation.

        Returns:
            Tuple[float, float]: Concordance Index (CI) and Integrated Brier Score (IBS).
        """
        survival_test = pd.DataFrame()
        survival_test['event'] = df_gt['failure'].astype(bool)
        survival_test['duration'] = df_gt['duration']

        lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])

        survival_estim = df_pred.drop(['serial_number', 'time'], axis='columns')
        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times
        )
        return ci, ibs

    def get_ci_and_ibs_agg(self, model, df_pred: pd.DataFrame, df_gt: pd.DataFrame, times: np.ndarray) -> Tuple[float, float]:
        """Calculate Concordance Index (CI) and Integrated Brier Score (IBS).

        Args:
            model: The trained model used for predictions.
            df_pred (pd.DataFrame): DataFrame containing predicted survival functions.
            df_gt (pd.DataFrame): DataFrame containing ground truth durations and event indicators.
            times (np.ndarray): Array of time points for evaluation.

        Returns:
            Tuple[float, float]: Concordance Index (CI) and Integrated Brier Score (IBS).
        """
        survival_test = pd.DataFrame()
        survival_test['event'] = df_gt['failure'].astype(bool)
        survival_test['duration'] = df_gt['duration']

        lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])

        survival_estim = df_pred.drop(['serial_number', 'time'], axis='columns')
        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times
        )
        return ci, ibs

    def bal_ibs_remain(self, survival_train, survival_test, estimate, times, axis=-1):
        """ IBS with equal impact of each event type and partial observation with controlled quantity """
        ibs_event = ibs_remain(survival_train, survival_test[survival_test["event"]],
                               estimate[survival_test["event"]], times, axis=axis)
        ibs_cens = ibs_remain(survival_train, survival_test[~survival_test["event"]],
                              estimate[~survival_test["event"]], times, axis=axis)
        return (ibs_event + ibs_cens)/2

    def get_ci_ibs_ibs_bal(self, model, df_pred: pd.DataFrame, df_gt: pd.DataFrame, times: np.ndarray, axis=-1) -> Tuple[float, float, float]:
        """Calculate Concordance Index (CI) and Integrated Brier Score (IBS).

        Args:
            model: The trained model used for predictions.
            df_pred (pd.DataFrame): DataFrame containing predicted survival functions.
            df_gt (pd.DataFrame): DataFrame containing ground truth durations and event indicators.
            times (np.ndarray): Array of time points for evaluation.

        Returns:
            Tuple[float, float, float]: Concordance Index (CI), Integrated Brier Score (IBS) and balanced IBS.
        """
        survival_test = pd.DataFrame()
        survival_test['event'] = df_gt['failure'].astype(bool)
        survival_test['duration'] = df_gt['duration']

        lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

        ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])

        survival_estim = df_pred.drop(['serial_number', 'time'], axis='columns')
        ibs = ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times,
            axis=axis
        )

        ibs_bal = self.bal_ibs_remain(
            None,
            survival_test.to_records(index=False),
            survival_estim,
            times,
            axis=axis
        )

        return ci, ibs, ibs_bal

    def get_metrics(self, model, df_pred: pd.DataFrame, df_gt: pd.DataFrame, times: np.ndarray, metrics: Set[str], axis=-1, df_train=None):

        metrics_dict = {}

        if df_train is not None:
            iauc_train = pd.DataFrame()
            iauc_train['cens'] = df_train['failure'].astype(bool)
            iauc_train['duration'] = df_train['duration']

        survival_test = pd.DataFrame()
        survival_test['event'] = df_gt['failure'].astype(bool)
        survival_test['duration'] = df_gt['duration']

        iauc_test = pd.DataFrame()
        iauc_test['cens'] = df_gt['failure'].astype(bool)
        iauc_test['time'] = df_gt['duration']

        lifetime_pred = model.get_expected_time_by_predictions(df_pred, times)

        if 'ci' in metrics:
            ci = concordance_index(df_gt['duration'], lifetime_pred, df_gt['failure'])
            metrics_dict['ci'] = ci

        survival_estim = df_pred.drop(['serial_number', 'time'], axis='columns')

        if 'ibs' in metrics:
            ibs = ibs_remain(
                None,
                survival_test.to_records(index=False),
                survival_estim,
                times,
                axis=axis
            )
            metrics_dict['ibs'] = ibs
        if 'ibs_bal' in metrics:

            ibs_bal = self.bal_ibs_remain(
                None,
                survival_test.to_records(index=False),
                survival_estim,
                times,
                axis=axis
            )
            metrics_dict['ibs_bal'] = ibs_bal

        if 'iauc' in metrics:
            if df_train is None:
                raise ValueError("df_train must be provided to compute iauc")
            hazard_estim = -np.log(survival_estim)
            iauc_score = iauc_WW_TI(
                iauc_train.to_records(index=False),
                iauc_test.to_records(index=False),
                hazard_estim,
                times,
            )
            metrics_dict['iauc'] = iauc_score

        return metrics_dict
