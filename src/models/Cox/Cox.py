import pandas as pd
import numpy as np
import torch
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
from torch.utils.data import DataLoader
from tqdm import tqdm


class CoxTimeVaryingEstimator(CoxTimeVaryingFitter):


    def __init__(self, penalizer=0.0, l1_ratio=0.0, event_col="failure", time_col='time', id_col='serial_number', device=None):
        super().__init__(penalizer=penalizer, l1_ratio=l1_ratio)
        self.event_col = event_col
        self.time_col = time_col
        self.id_col = id_col
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None

    def _batch_to_df(self, batch):
        serial_numbers, obs_times, X, y, durations = batch
        serial_numbers = np.array(serial_numbers)
        obs_times = np.array(obs_times)
        y = np.array(y).astype(int)
        durations = np.array(durations)
        features = X.cpu().numpy() if torch.is_tensor(X) else np.array(X)
        df = pd.DataFrame(features)
        df[self.id_col] = serial_numbers
        df[self.time_col] = obs_times
        df[self.event_col] = y 
        df['duration'] = durations
        return df

    def fit(self, train_dataloader: DataLoader):
        dfs = []
        for batch in tqdm(train_dataloader, desc="Collecting data for Cox fit"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)
        df_tv = self._to_start_stop(df_all)
        df_to_fit = df_tv.drop(columns=[self.time_col])
        super().fit(df_to_fit, id_col=self.id_col, start_col='start', stop_col='stop', event_col=self.event_col)
        self.feature_cols = [c for c in df_tv.columns if c not in [
            self.id_col, 'start', 'stop', self.event_col, 'duration']]
        return self

    def _to_start_stop(self, df):
        """
        Creates DataFrame for CoxTimeVaryingFitter:
        - start = time - min_time_in_group  
        - stop = start of next observation, for last: start + duration
        - event_col for all except last: 0 (censored), for last: current value
        """
        df = df.sort_values([self.id_col, self.time_col]).copy()
        min_times = df.groupby(self.id_col)[self.time_col].transform('min')
        df['start'] = df[self.time_col] - min_times
        df['stop'] = df.groupby(self.id_col)['start'].shift(-1)
        last_mask = df['stop'].isna()
        df.loc[last_mask, 'stop'] = df.loc[last_mask, 'start'] + df.loc[last_mask, 'duration']
        df.loc[~last_mask, self.event_col] = 0  # All observations except last are censored
        df = df.drop(columns=['duration'])
        return df

    def _get_survival_function(self, X_features, times):
        baseline_surv = self.baseline_survival_
        fill_indices = baseline_surv.index.searchsorted(times, side='right') - 1
        fill_indices = np.clip(fill_indices, 0, max(fill_indices) - 1)
        baseline_surv_interp = baseline_surv.iloc[fill_indices, 0].values
        partial_haz = self.predict_partial_hazard(X_features).values
        surv = baseline_surv_interp[None, :] ** partial_haz[:, None]
        return surv

    def predict(self, dataloader: DataLoader, times: np.ndarray):
        """
        Collects all data from DataLoader into one DataFrame, then predicts survival functions for all at once.
        Returns: (df_surv, df_gt)
        """
        # First collect the entire dataset into one DataFrame
        dfs = []
        for batch in tqdm(dataloader, desc="Collecting data for Cox prediction"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)

        X_feat = df_all[self.feature_cols]
        times = np.array(times)
        surv = self._get_survival_function(X_feat, times)
        pred_values = np.column_stack([df_all[self.time_col].values, surv])
        serial_numbers_flat = df_all[self.id_col].values
        columns = ['time'] + times.tolist()
        df_surv = pd.DataFrame(pred_values, columns=columns)
        df_surv.insert(0, 'serial_number', serial_numbers_flat)
        df_surv['time'] = df_surv['time'].astype('int32')

        # gt (ground truth)
        if 'duration' in df_all:
            gt_values = np.column_stack([
                df_all[self.time_col].values,
                df_all['duration'].values,
                df_all[self.event_col].values
            ])
            df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
            df_gt.insert(0, 'serial_number', serial_numbers_flat)
            df_gt = df_gt.astype({'serial_number': 'string', 'time': 'int32', 'duration': 'int32'})
            df_gt['failure'] = df_gt['failure'] == 1
        else:
            df_gt = pd.DataFrame()

        return df_surv, df_gt

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray):
        df_surv, df_gt = self.predict(dataloader, times)
        return self.get_expected_time_by_predictions(df_surv, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray):
        survival_vec = X_pred.drop(['serial_number', 'time'], axis='columns').values
        return np.trapz(y=survival_vec, x=times)


class CoxTimeInvariantSNFitter(CoxPHFitter):

    def __init__(self, penalizer=0.0, l1_ratio=0.0, event_col="failure", time_col='time', id_col='serial_number', device=None):
        super().__init__(penalizer=penalizer, l1_ratio=l1_ratio)
        self.event_col = event_col
        self.time_col = time_col
        self.id_col = id_col
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None

    def _batch_to_df(self, batch):
        serial_numbers, obs_times, X, y, durations = batch
        serial_numbers = np.array(serial_numbers)
        obs_times = np.array(obs_times)
        y = np.array(y).astype(int)
        durations = np.array(durations)
        features = X.cpu().numpy() if torch.is_tensor(X) else np.array(X)
        df = pd.DataFrame(features)
        df[self.id_col] = serial_numbers
        df[self.time_col] = obs_times
        df[self.event_col] = y
        df['duration'] = durations
        return df

    def fit(self, train_dataloader: DataLoader):
        dfs = []
        for batch in tqdm(train_dataloader, desc="Collecting data for Cox fit"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)
        df_tv = self._to_start_stop(df_all)
        df_to_fit = df_tv.drop(columns=[self.time_col, self.id_col])
        super().fit(df_to_fit, duration_col='duration', event_col=self.event_col)
        self.feature_cols = [c for c in df_tv.columns if c not in [
            self.id_col, self.time_col, self.event_col, 'duration']]
        return self

    def _to_start_stop(self, df):
        """
        Prepares DataFrame for CoxPHFitter by keeping only the first observation for each subject.
        This is appropriate for time-invariant Cox models where we use baseline covariates only.
        """
        df = df.sort_values([self.id_col, self.time_col]).copy()
        df = df[(df['time'] == df.groupby(self.id_col)['time'].transform('min'))]
        return df

    def _get_survival_function(self, X_features, times):
        baseline_surv = self.baseline_survival_
        fill_indices = baseline_surv.index.searchsorted(times, side='right') - 1
        fill_indices = np.clip(fill_indices, 0, max(fill_indices) - 1)
        baseline_surv_interp = baseline_surv.iloc[fill_indices, 0].values
        partial_haz = self.predict_partial_hazard(X_features).values
        surv = baseline_surv_interp[None, :] ** partial_haz[:, None]
        return surv

    def predict(self, dataloader: DataLoader, times: np.ndarray):
        """
        Collects all data from DataLoader into one DataFrame, then predicts survival functions for all at once.
        Returns: (df_surv, df_gt)
        """
        # First collect the entire dataset into one DataFrame
        dfs = []
        for batch in tqdm(dataloader, desc="Collecting data for Cox prediction"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)

        X_feat = df_all[self.feature_cols]
        times = np.array(times)
        surv = self._get_survival_function(X_feat, times)
        pred_values = np.column_stack([df_all[self.time_col].values, surv])
        serial_numbers_flat = df_all[self.id_col].values
        columns = ['time'] + times.tolist()
        df_surv = pd.DataFrame(pred_values, columns=columns)
        df_surv.insert(0, 'serial_number', serial_numbers_flat)
        df_surv['time'] = df_surv['time'].astype('int32')

        # gt (ground truth)
        if 'duration' in df_all:
            gt_values = np.column_stack([
                df_all[self.time_col].values,
                df_all['duration'].values,
                df_all[self.event_col].values
            ])
            df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
            df_gt.insert(0, 'serial_number', serial_numbers_flat)
            df_gt = df_gt.astype({'serial_number': 'string', 'time': 'int32', 'duration': 'int32'})
            df_gt['failure'] = df_gt['failure'] == 1
        else:
            df_gt = pd.DataFrame()

        return df_surv, df_gt

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray):
        df_surv, df_gt = self.predict(dataloader, times)
        return self.get_expected_time_by_predictions(df_surv, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray):
        survival_vec = X_pred.drop(['serial_number', 'time'], axis='columns').values
        return np.trapz(y=survival_vec, x=times)


class CoxTimeInvariantLNFitter(CoxPHFitter):

    def __init__(self, penalizer=0.0, l1_ratio=0.0, event_col="failure", time_col='time', id_col='serial_number', device=None):
        super().__init__(penalizer=penalizer, l1_ratio=l1_ratio)
        self.event_col = event_col
        self.time_col = time_col
        self.id_col = id_col
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None

    def _batch_to_df(self, batch):
        serial_numbers, obs_times, X, y, durations = batch
        serial_numbers = np.array(serial_numbers)
        obs_times = np.array(obs_times)
        y = np.array(y).astype(int)
        durations = np.array(durations)
        features = X.cpu().numpy() if torch.is_tensor(X) else np.array(X)
        df = pd.DataFrame(features)
        df[self.id_col] = serial_numbers
        df[self.time_col] = obs_times
        df[self.event_col] = y
        df['duration'] = durations
        return df

    def fit(self, train_dataloader: DataLoader):
        dfs = []
        for batch in tqdm(train_dataloader, desc="Collecting data for Cox fit"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)
        df_to_fit = df_all.drop(columns=[self.time_col, self.id_col])
        super().fit(df_to_fit, duration_col='duration', event_col=self.event_col)
        self.feature_cols = [c for c in df_all.columns if c not in [
            self.id_col, self.time_col, self.event_col, 'duration']]
        return self

    def _get_survival_function(self, X_features, times):
        baseline_surv = self.baseline_survival_
        fill_indices = baseline_surv.index.searchsorted(times, side='right') - 1
        fill_indices = np.clip(fill_indices, 0, max(fill_indices) - 1)
        baseline_surv_interp = baseline_surv.iloc[fill_indices, 0].values
        partial_haz = self.predict_partial_hazard(X_features).values
        surv = baseline_surv_interp[None, :] ** partial_haz[:, None]
        return surv

    def predict(self, dataloader: DataLoader, times: np.ndarray):
        """
        Collects all data from DataLoader into one DataFrame, then predicts survival functions for all at once.
        Returns: (df_surv, df_gt)
        """
        # First collect the entire dataset into one DataFrame
        dfs = []
        for batch in tqdm(dataloader, desc="Collecting data for Cox prediction"):
            batch_df = self._batch_to_df(batch)
            dfs.append(batch_df)
        df_all = pd.concat(dfs, ignore_index=True)

        X_feat = df_all[self.feature_cols]
        times = np.array(times)
        surv = self._get_survival_function(X_feat, times)
        pred_values = np.column_stack([df_all[self.time_col].values, surv])
        serial_numbers_flat = df_all[self.id_col].values
        columns = ['time'] + times.tolist()
        df_surv = pd.DataFrame(pred_values, columns=columns)
        df_surv.insert(0, 'serial_number', serial_numbers_flat)
        df_surv['time'] = df_surv['time'].astype('int32')

        # gt (ground truth)
        if 'duration' in df_all:
            gt_values = np.column_stack([
                df_all[self.time_col].values,
                df_all['duration'].values,
                df_all[self.event_col].values
            ])
            df_gt = pd.DataFrame(gt_values, columns=['time', 'duration', 'failure'])
            df_gt.insert(0, 'serial_number', serial_numbers_flat)
            df_gt = df_gt.astype({'serial_number': 'string', 'time': 'int32', 'duration': 'int32'})
            df_gt['failure'] = df_gt['failure'] == 1
        else:
            df_gt = pd.DataFrame()

        return df_surv, df_gt

    def get_expected_time(self, dataloader: DataLoader, times: np.ndarray):
        df_surv, df_gt = self.predict(dataloader, times)
        return self.get_expected_time_by_predictions(df_surv, times), df_gt

    def get_expected_time_by_predictions(self, X_pred: pd.DataFrame, times: np.ndarray):
        survival_vec = X_pred.drop(['serial_number', 'time'], axis='columns').values
        return np.trapz(y=survival_vec, x=times)
