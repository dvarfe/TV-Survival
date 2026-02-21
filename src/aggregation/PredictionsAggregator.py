import numpy as np
import pandas as pd


class PredictionsAggregator():

    def __init__(self, mode, weight=None):
        """
        Теперь будем делать только агрегацию предсказаний по датафрейму с уже готовыми предсказаниями
        """

        self.mode = mode
        self.id_col = 'serial_number'
        self.time_col = 'time'
        self.event_col = 'failure'
        self.EPS = 0.0001

        if (not mode in ['n_dist', 't_dist', 'geom', 'prob_dist', 'last', 'first']):
            raise ValueError('Wrong mode!')

        # Задаём параметры по умолчанию
        if mode == 'n_dist' and (weight is None):
            self.weight = 1
        elif mode == 'n_dist':
            self.weight = weight
        if mode == 't_dist' and (weight is None):
            self.scale = 1
        elif mode == 't_dist':
            self.scale = weight
        if mode == 'geom' and (weight is None):
            self.alpha = 0.5
        elif mode == 'geom':
            self.alpha = weight

    def get_expected_time_by_predictions(self, X_pred, times=None):
        survival_vec = X_pred.drop([self.id_col, self.time_col], axis='columns')
        return np.trapz(y=survival_vec, x=times if not times is None else survival_vec.columns)

    def get_extended_times(self, X, times):
        timeshift = X.groupby(self.id_col)[self.time_col].transform('max') - X[self.time_col]
        times_extended = self.extend_times(times, max(timeshift))
        return times_extended

    def predict(self, X_pred, times, timeshift):
        '''
        X_pred - предсказание, которые было получено по times_extended!
        times - параметр-заглушка, нужен для совместимости с ibs_TV, в будущем надо будет избавиться
        weight - Параметр, который отвечает за вклад старых наблюдений в n_dist. Должен быть < 1.
        scale - параметр, который отвечает за вклад старых наблюдений в t_dist. В разных наборах данных
        может быть разное время между наблюдениями, где-то показания снимаются каждые 5 единиц времени, а где-то
        каждые 25. Так как метод t_dist основан на разнице во времени между наблюдениями, необходимо учитывать это.
        Так как методы являются взаимоисключающими, только один параметр может быть не-None.
        '''
        mode = self.mode

        if times is None:
            raise ValueError('Times must be non-None!')

        pred_corrected = self.correct_predictions(X_pred, max(times), np.array(timeshift))

        if mode == 'n_dist':
            pred_corrected.drop('surv_prob', axis='columns', inplace=True)
            return self.n_dist_agg(pred_corrected)

        if mode == 't_dist':
            pred_corrected.drop('surv_prob', axis='columns', inplace=True)
            return self.t_dist_agg(pred_corrected)

        if mode == 'geom':
            pred_corrected.drop('surv_prob', axis='columns', inplace=True)
            return self.geom_agg(pred_corrected)

        if mode == 'prob_dist':
            return self.prob_dist_agg(pred_corrected)

        if mode == 'last':
            pred_corrected.drop('surv_prob', axis='columns', inplace=True)
            return self.last_agg(pred_corrected)

        if mode == 'first':
            pred_corrected.drop('surv_prob', axis='columns', inplace=True)
            return self.first_agg(pred_corrected)

    def extend_times(self, times, max_shift):
        '''
        Принимает на вход шкалу times и максимальный сдвиг относительно свежайшего наблюдения.
        Возвращает шкалу от 0 до max(times) + max_shift
        '''
        return np.arange(0, max_shift + max(times) + 1)

    def correct_predictions(self, X, max_times, timeshift):
        '''
        Выравниваем наблюдения по самому актуальному.
        X - данные
        max_times - длина times
        timeshift - положительная величина. Количество t, на которое наблюдение устарело относительно 
        самого последнего.
        '''
        X_res = X.drop([self.id_col, self.time_col], axis='columns').values
        row_indices = np.arange(X.shape[0])[:, None]
        col_indices = np.arange(max_times + 1) + timeshift[:, None]

        X_res = X_res[row_indices, col_indices]
        # predictions_end = np.arange(X_res.shape[1]) >= timeshift[:, np.newaxis]
        predictions_end = np.arange(X_res.shape[1]) >= X_res.shape[1] - timeshift[:, np.newaxis]
        X_res[predictions_end] = 0

        surv_prob = np.clip(X_res[:, 0], self.EPS, None)
        X_res = X_res / surv_prob[:, None]

        X_df = pd.DataFrame(X_res, columns=np.arange(max_times + 1), index=X.index)
        X_df[self.id_col] = X[self.id_col]
        X_df[self.time_col] = X[self.time_col]
        X_df['surv_prob'] = surv_prob
        return X_df

    def n_dist_agg(self, X):
        '''
        Осуществляет агрегацию наблюдений, при которой предсказания с меньшим порядковым номером(более старые)
        дают меньший вклад
        X - датафрейм, который содержит значения функции выживания для различных наблюдений, 
            имеет колонки id_col и time_col  
        '''

        result_df = X.copy().sort_values(by=self.time_col, ascending=False)

        # Определяем степень удалённости n_dist для каждого наблюдения
        n_dist_values = result_df.groupby(self.id_col).cumcount(ascending=True)

        # Рассчитываем веса каждого ряда наблюдений
        result_df['weights'] = self.weight ** n_dist_values

        # Оставляем только колонки, которые отвечают за вероятности
        time_columns = result_df.columns[(result_df.columns != self.id_col) & (
            result_df.columns != 'weights') & (result_df.columns != self.time_col)].tolist()

        # Домножаем каждую строку на соответствующий вес
        result_df.loc[:, time_columns] = result_df.loc[:, time_columns].multiply(
            result_df['weights'], axis=0).astype('float32')

        # Вычисляем взвешенное среднее
        result_df = result_df.groupby(self.id_col)[time_columns].sum().div(
            result_df.groupby(self.id_col)['weights'].sum(), axis='rows')
        # Создаём итоговую строку с id
        result_df.loc[:, self.time_col] = X.groupby(self.id_col)[self.time_col].max()
        result_df = result_df.reset_index(names=self.id_col)
        return result_df

    def t_dist_agg(self, X):
        '''
        Осуществляет агрегацию наблюдений, при которой предсказания соответствующие более старым наблюдениям дают меньший вклад.
        Старость определяется как время от самого актуального события
        X - датафрейм, который содержит значения функции выживания для различных наблюдений, 
            имеет колонки id_col и time_col  
        '''
        X_copy = X.sort_values(by=self.time_col, ascending=True)

        # Определяем степень удалённости t_dist для каждого наблюдения
        t_dist_values = X_copy[self.time_col] - X_copy.groupby(self.id_col)[self.time_col].transform('max')

        # Рассчитываем веса каждого ряда наблюдений
        X_copy['weights'] = np.exp(t_dist_values/self.scale)

        # Оставляем только колонки, которые отвечают за вероятности
        time_columns = X_copy.columns[(X_copy.columns != self.id_col) & (
            X_copy.columns != self.time_col) & (X_copy.columns != 'weights')].tolist()
        numeric_data = X_copy[time_columns]

        # Домножаем каждую строку на соответствующий вес
        X_copy.loc[:, time_columns] = numeric_data.mul(X_copy['weights'], axis='rows').astype('float32')

        # Вычисляем взвешенное среднее
        aggregated = X_copy.groupby(self.id_col).sum().div(X_copy.groupby(self.id_col)['weights'].sum(), axis='rows')

        # Создаём итоговую строку с id
        aggregated.loc[:, self.time_col] = X.groupby(self.id_col)[self.time_col].max()
        result_df = aggregated.reset_index(names=self.id_col)
        result_df = result_df.drop('weights', axis='columns')
        return result_df

    def geom_agg(self, X):
        '''
        Осуществляет геометрическую агрегацию.
        X - датафрейм, который содержит значения функции выживания для различных наблюдений, 
            имеет колонки id_col и time_col  
        '''
        result_df = X.copy().sort_values(by=self.time_col, ascending=False)
        powers = result_df.groupby(self.id_col).cumcount(ascending=True)
        result_df['weights'] = (1 - self.alpha) ** powers
        result_df.loc[powers != result_df.groupby(self.id_col).transform('size') - 1, 'weights'] *= self.alpha

        time_columns = result_df.drop([self.id_col, self.time_col, 'weights'], axis='columns').columns
        result_df.loc[:, time_columns] = result_df.loc[:, time_columns].mul(
            result_df['weights'], axis='rows').astype('float32')
        result_df = result_df.groupby(self.id_col)[time_columns].sum().div(
            result_df.groupby(self.id_col)['weights'].sum(), axis='rows')

        result_df.loc[:, self.time_col] = X.groupby(self.id_col)[self.time_col].max()
        result_df = result_df.reset_index(names=self.id_col)
        # result_df.loc[:,self.time_col] = 0 # заглушка, потому что это колонка не используется, в будущем можно сделать настоящей

        return result_df

    def prob_dist_agg(self, X):
        '''
        Осуществляет агрегацию, основанную на значении функции выживания.
        X - датафрейм, который содержит значения функции выживания для различных наблюдений, 
            колонки id_col и time_col, а также колонку surv_prob, в которой хранится значение
            вероятности выживания данного наблюдения, в момент самого актуального наблюдения группы. 
        '''
        time_columns = X.drop([self.id_col, self.time_col, 'surv_prob'], axis='columns').columns
        pred_mul = X.copy()
        pred_mul.loc[:, time_columns] = pred_mul.loc[:, time_columns].mul(pred_mul['surv_prob'], axis='rows')
        result_df = pred_mul.groupby(self.id_col)[time_columns].sum().div(
            pred_mul.groupby(self.id_col)['surv_prob'].sum(), axis='rows')

        result_df.loc[:, self.time_col] = X.groupby(self.id_col)[self.time_col].max()
        result_df = result_df.reset_index(names=self.id_col)
        # result_df.loc[:,self.time_col] = X[X.groupby(self.id_col)[self.time_col].transform('max') == X[self.time_col]][self.time_col].reset_index(drop=True)

        return result_df

    def last_agg(self, X):
        return X[X[self.time_col] == X.groupby(self.id_col)[self.time_col].transform('max')]

    def first_agg(self, X):
        return X[X[self.time_col] == X.groupby(self.id_col)[self.time_col].transform('min')]


class HazardSumAgg():
    def __init__(self, serial_number_col='serial_number', time_col='time', event_col='failure'):
        """
        Отдельный класс для прогнозирования по схеме с суммированием рисков/перемножением вероятностей
        """

        self.id_col = serial_number_col
        self.time_col = time_col
        self.event_col = event_col
        self.EPS = 0.0001

    def get_expected_time_by_predictions(self, X_pred, times=None):
        survival_vec = X_pred.drop([self.id_col, self.time_col], axis='columns')
        return np.trapz(y=survival_vec, x=times if not times is None else survival_vec.columns)

    def get_extended_times(self, X, times):
        '''Здесь формула простая, из-за того, что прогнозы сдвигаются не так, как при агрегации'''
        times_extended = np.arange(min(times), max(times) + 1)
        return times_extended

    def get_timeshift(self, X):
        return X[self.time_col] - X.groupby(self.id_col)[self.time_col].transform('min')

    def predict(self, X_pred, times, timeshift):
        '''
        X_pred - предсказание, которые было получено по times_extended!
        '''
        if times is None:
            raise ValueError('Times must be non-None!')

        pred_corrected = self.correct_predictions(X_pred, max(times), np.array(timeshift))
        return self.agg(pred_corrected)

    def correct_predictions(self, X, max_times, timeshift):
        """
        Сдвигаем прогнозы вправо на timeshift, заполняя пустые ячейки единицами.

        Параметры:
            X          - входные данные (DataFrame)
            max_times  - максимальное количество временных точек (int)
            timeshift  - массив смещений для каждой строки (np.array)
        """
        # Извлекаем только значения признаков (без id и time)
        X_val = X.drop([self.id_col, self.time_col], axis='columns').values
        n_rows, n_cols = X_val.shape

        # Заполним массив единицами - "нейтральное" значение S(t).
        X_res = np.ones((n_rows, max_times + 1))

        # Здесь без цикла не обойтись. Каждая строка имеет разный размер.
        for i in range(n_rows):
            shift = timeshift[i]
            end = min(n_cols, max_times + 1 - shift)
            X_res[i, shift:shift + end] = X_val[i, :end]

        # Возвращаем DataFrame
        X_df = pd.DataFrame(X_res, columns=np.arange(max_times + 1), index=X.index)
        X_df[self.id_col] = X[self.id_col]
        X_df[self.time_col] = X[self.time_col]
        return X_df

    def agg(self, X):
        X_res = X.drop(self.time_col, axis='columns')
        X_res = X_res.groupby(self.id_col).prod()
        X_res[self.time_col] = X.groupby(self.id_col)[self.time_col].transform('min')
        return X_res
