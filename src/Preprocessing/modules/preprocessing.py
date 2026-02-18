import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# This class transforms time to format {0, 1, ...}, renames time column
# and keeps only features that were present during training
class TimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='id', time_column='time', event_column='event', inplace=False, rename=True):
        '''
        Parameters:
        id_col - Name of ID column
        time_column - Name of time column
        event_column - Name of event column
        inplace - Whether to perform transformations in place
        rename - Whether to rename time column to 'time'
        '''
        self.id_col = id_col
        self.time_column = time_column
        self.event_column = event_column
        self.inplace = inplace
        self.rename = rename

    def fit(self, X, y=None):
        self.features = set(X.columns).difference([self.id_col, self.time_column, self.event_column])
        return self

    def transform(self, X, y=None):
        # Choose between in-place modification or creating a copy
        X_copy = X if self.inplace else X.copy()

        # Convert to datetime
        X_copy[self.time_column] = pd.to_datetime(X_copy[self.time_column])
        # Transform time scale to start from 0 for each ID
        X_copy[self.time_column] = X_copy[self.time_column] - \
            X_copy.groupby(self.id_col)[self.time_column].transform('min')
        X_copy[self.time_column] = X_copy[self.time_column].dt.days.astype(int)
        X_copy = X_copy.drop(columns=X_copy.columns[~X_copy.columns.isin(list(self.features.union(
            [self.id_col, self.time_column, self.event_column])))])

        X_copy['max_lifetime'] = X_copy.groupby(self.id_col)[self.time_column].transform('max')
        if self.event_column in X_copy.columns:
            X_copy.loc[:, self.event_column] = X_copy.groupby(self.id_col)[self.event_column].transform('max')

        if self.rename:
            if self.event_column in X_copy.columns:
                X_copy = X_copy.rename(columns={self.time_column: 'time',
                                                self.id_col: 'serial_number', self.event_column: 'failure'})
            else:
                X_copy = X_copy.rename(columns={self.time_column: 'time',
                                                self.id_col: 'serial_number'})

        return X_copy

# Transform that renames event column to 'event' and id column to 'id'
# for consistency, and removes observations with >1 event


class InitTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, event_column='event', id_column='id', inplace=False):
        '''
        Parameters:
        event_column - Name of event label column
        id_column - Name of object identifier column  
        inplace - Whether to perform transformations in place
        '''
        self.event_column = event_column
        self.id_column = id_column
        self.inplace = inplace

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Choose between in-place modification or creating a copy
        X_copy = X if self.inplace else X.copy()

        # Get event counts for each ID
        X_events = X_copy[[self.id_column, self.event_column]].groupby('id').sum()

        # Get IDs with more than 1 event
        drop_categories = X_events[X_events[self.event_column] > 1].index

        # Remove rows with selected IDs
        X_copy = X_copy.drop(X_copy[X_copy[self.id_column].isin(drop_categories)].index).reset_index(drop=True)

        return X_copy

# Class that renames dataframe columns to id, event, time
# Optional to use


class ColRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, event_column='failure', id_column='serial_number',
                 time_column='time', inplace=False):
        '''
        Parameters:
        event_column - Name of event label column
        id_column - Name of object identifier column
        time_column - Name of time column
        inplace - Whether to perform transformations in place
        '''
        self.event_column = event_column
        self.id_column = id_column
        self.time_column = time_column
        self.inplace = inplace

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Choose between in-place modification or creating a copy
        X_copy = X if self.inplace else X.copy()

        X_copy = X_copy.rename(columns={self.event_column: 'event',
                                        self.id_column: 'id',
                                        self.time_column: 'time'})

        return X_copy

# Class for removing truncated observations


class TruncRemover(BaseEstimator, TransformerMixin):
    def __init__(self, inplace=False, time_col='time', id_col='id', event_col='event'):
        self.inplace = inplace
        self.time_col = time_col
        self.id_col = id_col
        self.event_col = event_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X if self.inplace else X.copy()
        last_observation = X_copy[self.time_col].max()
        trunc_id = X_copy[(X_copy[self.time_col] == last_observation) &
                          (X_copy[self.event_col] != 1)][self.id_col].unique()
        X_copy.drop(X_copy[X_copy[self.id_col].isin(trunc_id)].index, inplace=True)
        return X_copy

# Class that allows sampling subsets from given dataset


class Sampler():
    def __init__(self, df, random_state=42, mode='random', time_col='time', id_col='id'):
        '''
        Parameters:
        mode - Sampling mode, options: 'random', 'first'
        '''
        self.time_col = time_col
        self.id_col = id_col

        # self.shuffled_df_idx - GroupBy object containing objects grouped by id

        if mode == 'random':
            # Keep only intermediate events for each object (not first and not last)
            self.shuffled_df_idx = (df[~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('max')) &
                                       ~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('min'))]
                                    .sample(frac=1, random_state=random_state).
                                    # Keep only id column and group (we only need indices for fixed permutation)
                                    loc[:, [self.id_col]].
                                    groupby(self.id_col))
        elif mode == 'first':
            df = df.sort_values(by=self.time_col)
            self.shuffled_df_idx = (df[~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('max')) &
                                       ~(df[self.time_col] == df.groupby(self.id_col)[self.time_col].transform('min'))]
                                    .loc[:, [self.id_col]]
                                    .groupby(self.id_col))
        self.X = df.copy()

    def get_n_samples(self, n_samples=10):
        # Get n intermediate observations
        df_sampled = self.X.loc[self.shuffled_df_idx.head(n_samples - 1).index, :]
        # Last record in each observation chain should reflect the final state
        df_sampled = pd.concat([df_sampled,
                                self.X[
                                    (self.X[self.time_col] == self.X.groupby(self.id_col)[self.time_col].transform('max')) |
                                    (self.X[self.time_col] == self.X.groupby(
                                        self.id_col)[self.time_col].transform('min'))
                                ]])  # Add first and last observations to intermediate ones
        return df_sampled


# Class for removing features with high number of missing values
# or features specified in features_to_remove
class FeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, nan_fraction=0.5, features_to_remove=None, event_col='failure', inplace=False):
        '''
        Parameters:
        nan_fraction - Acceptable fraction of missing values
        features_to_remove - List of feature names that must be removed
        '''
        self.event_col = event_col
        self.features_to_remove = features_to_remove
        self.nan_fraction = nan_fraction
        self.inplace = inplace

    def fit(self, X, y=None):
        na_count = X.isna().sum()
        high_na = [
            c for c in X.columns
            if na_count[c] > X.shape[0] * self.nan_fraction
        ]
        self.features_to_remove_ = set((self.features_to_remove or []) + high_na + [self.event_col])
        self.features = list(set(X.columns).difference(self.features_to_remove_))
        return self

    def transform(self, X, y=None):
        X_copy = X if self.inplace else X.copy()
        features = (self.features + [self.event_col]) if self.event_col in X_copy.columns else self.features
        return X_copy[features]

# Class that aggregates observations that occurred at the same time point


class ObservationAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='id', time_col='time', inplace=False):
        self.inplace = inplace
        self.id_col = id_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Choose between in-place modification or creating a copy
        X_copy = X if self.inplace else X.copy()

        X_copy = X_copy.drop_duplicates(subset=[self.id_col, self.time_col], keep='last')

        return X_copy

# Class for filling missing values in data. For test set, we'll likely fill with constants.


class NanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_val=0, id_col='id', time_col='time', inplace=False):
        '''
        Parameters:
        fill_val - Default value for filling missing values.
                  Used if the very first observation contains NaN.
        '''
        self.inplace = inplace
        self.fill_val = fill_val
        self.id_col = id_col
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Choose between in-place modification or creating a copy
        X_copy = X if self.inplace else X.copy()

        # Sort to ensure bfill works correctly
        X_copy = X_copy.sort_values(by=[self.id_col, self.time_col])

        # Apply bfill for each ID
        X_copy.loc[:, X_copy.columns != self.id_col] = (X_copy
                                                        .groupby(self.id_col)
                                                        .transform('bfill')
                                                        .infer_objects(copy=False)
                                                        .fillna(self.fill_val)
                                                        )

        return X_copy

# Function for splitting dataset into training and validation sets


def stratified_split(df, test_size=0.2, random_state=42, id_col='id', event_col='event'):
    id_events = df.groupby(id_col)[event_col].max().reset_index()
    train_id, test_id = train_test_split(id_events[id_col], test_size=test_size, random_state=random_state,
                                         stratify=id_events[event_col])

    df_train = df[df[id_col].isin(train_id)]
    df_test = df[df[id_col].isin(test_id)]

    return df_train, df_test


class TimeMaskedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, features, id_col='id', time_col='time'):
        self.features = features
        self.id_col = id_col
        self.time_col = time_col
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = X.copy()
        mask = (
            X.groupby(self.id_col)[self.time_col].transform('max')
            != X[self.time_col]
        )
        self.features_ = list(set(self.features).intersection(X.columns))
        self.scaler.fit(X.loc[mask, self.features_])
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[:, self.features_] = self.scaler.transform(X[self.features_])
        return X
