from random import random

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class Calendar(object):
    def __init__(self, days, burned_by_calculations=0):
        self.days = days
        self.burned_by_calculations = burned_by_calculations
        self.available_days = self.days[burned_by_calculations:]
        # self.available_days = len(self.days) - burned_by_calculations

    def get_available_days(self, train_and_val_size=0):
        return self.available_days[train_and_val_size:]

    def get_available_days_range(self, train_and_val_size=0):
        start, end = self.available_days[train_and_val_size], self.available_days[-1]
        assert end > start
        return start, end

    def get_days_to(self, date_to, day_count_learn, day_count_validate):
        available_days_capped = self.available_days[self.available_days < date_to]
        assert (len(available_days_capped) > 0)
        available_days_capped = available_days_capped[-day_count_learn - day_count_validate:]
        assert (len(available_days_capped) > 0)
        date_from = available_days_capped[0]
        days_capped = self.days[self.days < date_from]
        assert (len(days_capped) > 0)
        days_capped = days_capped[-self.burned_by_calculations:]
        assert (len(days_capped) > 0)
        date_from_burned = days_capped[0]
        date_from_validation = available_days_capped[day_count_learn]
        return date_from_burned, date_from, date_from_validation, date_to

    def move_by_days(self, from_date, by_days):
        where_index = np.where(self.days == from_date)[0]
        if len(where_index) == 0:
            return None
        if where_index[0] + by_days >= len(self.days):
            return None
        return self.days[where_index[0] + by_days]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def generate_ema_weights(periods):
    assert periods > 2
    ar = np.hstack([np.array([0, 1]), np.zeros(periods - 1)])
    return np.flip(pd.DataFrame(ar).ewm(span=periods, adjust=False).mean()[1:].values).reshape(-1)


def compute_tech_sma(df_pivo_all, period):
    smas = np.ma.average(rolling_window(df_pivo_all.values.swapaxes(0, 1), period), axis=2).swapaxes(0, 1)
    relative_smas = smas[1:] / smas[:-1]
    pd_smas = pd.DataFrame(relative_smas, columns=df_pivo_all.columns, index=df_pivo_all.index[-relative_smas.shape[0]:])
    return pd_smas


def compute_tech_ema(df_pivo_all, period):
    weights = generate_ema_weights(period)
    emas = np.ma.average(rolling_window(df_pivo_all.values.swapaxes(0, 1), period), axis=2, weights=weights, returned=False).swapaxes(0, 1)
    #return pd.DataFrame(emas, columns=df_pivo_all.columns, index=df_pivo_all.index[-emas.shape[0]:])

    relative_emas = emas[1:] / emas[:-1]
    pd_emas = pd.DataFrame(relative_emas, columns=df_pivo_all.columns, index=df_pivo_all.index[-relative_emas.shape[0]:])
    return pd_emas


def compute_tech_rsi(df_pivo_all, period):
    change = df_pivo_all.values[1:] - df_pivo_all.values[:-1]
    gain = np.maximum(change, np.zeros(change.shape[1]))
    loss = np.maximum(-change, np.zeros(change.shape[1]))
    gain_average = np.ma.average(rolling_window(gain.swapaxes(0, 1), period), axis=2).swapaxes(0, 1)
    loss_average = np.ma.average(rolling_window(loss.swapaxes(0, 1), period), axis=2).swapaxes(0, 1)
    rs = gain_average / loss_average
    rsi = np.where(loss_average == 0., 100., 100.-(100./(1. + rs)))
    return pd.DataFrame(rsi, columns=df_pivo_all.columns, index=df_pivo_all.index[-rsi.shape[0]:])

def mark_best(data_array, ratio, SUBSAMPLE_SCORING_LINSPACE):
    n_elements = len(data_array)
    n_from_left = int(n_elements * ratio)
    n_from_right = int(n_elements * (1.0 - ratio))
    sorted_ords = np.argsort(data_array)
    result = data_array > data_array[sorted_ords[n_from_left-1]]
    if n_from_left + n_from_right < n_elements:
        result[sorted_ords[n_from_left]] = bool(random.getrandbits(1))
    scores = np.zeros(n_elements)
    np.put_along_axis(scores, sorted_ords, SUBSAMPLE_SCORING_LINSPACE, axis=0)
    # scores2 = SUBSAMPLE_SCORING_LINSPACE[sorted_ords]
    return (result, scores)

def normalize_source2(train_X, divide_constant):
    normalization_means = train_X.mean(axis=(0, 2)).reshape((1, -1, 1))
    normalization_stds = train_X.std(axis=(0, 2)).reshape((1, -1, 1))
    train_X = np.subtract(train_X, normalization_means)
    train_X = np.divide(train_X, normalization_stds) / divide_constant
    train_X = train_X.clip(-1., 1.)
    return train_X, normalization_means, normalization_stds

def normalize_additional(test_X, normalization_means, normalization_stds, NORMALIZATION_DIVIDE_CONSTANT):
    vali_X = np.subtract(test_X, normalization_means)
    vali_X = np.divide(vali_X, normalization_stds) / NORMALIZATION_DIVIDE_CONSTANT
    vali_X = vali_X.clip(-1., 1.)
    return vali_X

N_COMPONENTS = 3  # 3

def generate_windows(df_closes, SIZE_WINDOW, N_PAST_VARIABLES):
    for start_index in range(0, df_closes.shape[0] - SIZE_WINDOW):
        df_window = df_closes.iloc[start_index:start_index+SIZE_WINDOW]
        date_start, date_end = df_window.iloc[0].name, df_window.iloc[-1].name
        df_next_bd = df_closes.iloc[start_index+SIZE_WINDOW]
        df_last_bds = df_window.iloc[-N_PAST_VARIABLES:]
        yield (date_start, date_end, df_window, df_last_bds, df_next_bd)

def generate_data_pca_x(df_closes, SIZE_WINDOW, N_PAST_VARIABLES, N_COMPONENTS, MARKED_AS_BEST_RATIO):
    for date_start, date_end, df_window, df_last_bds, df_next_bd in generate_windows(df_closes, SIZE_WINDOW,
                                                                                     N_PAST_VARIABLES):
        # print(f'date_start: {date_start}, date_end: {date_end}')
        pca = PCA(n_components=N_COMPONENTS, svd_solver='randomized')  # pca = PCA(0.75)
        pca.fit(df_window)
        values_last_bd = df_last_bds.values
        principals = pca.transform(values_last_bd)

        residuums = np.broadcast_to(principals.reshape((10, 1, 3)), (10, 20, 3)) * np.broadcast_to(pca.components_.swapaxes(0, 1).reshape((1, 20, 3)), (10, 20, 3))
        #res.sum(axis=2)

        # df_closes_pca = pd.DataFrame(pca.inverse_transform(principals), columns=df_closes.columns,
        #                              index=df_last_bds.index)
        #
        # df_residuums = pd.DataFrame(values_last_bd - df_closes_pca.values, columns=df_window.columns,
        #                             index=df_last_bds.index)
        y_output, scores = mark_best(df_next_bd.values, MARKED_AS_BEST_RATIO)
        next_bd_date = df_next_bd.name

        # values_resi_min = df_residuums.values.min(axis=1)
        # values_resi_max = df_residuums.values.max(axis=1)
        # values_resi_std = df_residuums.values.std(axis=1)
        values_resi_mean = residuums.mean(axis=1).swapaxes(0, 1)
        # values_resi_median = np.median(df_residuums.values, axis=1)
        X, Y, metas = [], [], []
        for index, symbol in enumerate(df_last_bds.columns):
            meta = (symbol, next_bd_date, scores[index])
            # X_params = np.vstack([
            #     df_closes_pca.values[:, index],
            #     #  values_resi_std,
            #     df_residuums.values[:, index],
            #     values_resi_median,
            #     values_resi_mean,
            #     #  values_resi_min,
            # ])
            X_params = residuums[:, index, :].swapaxes(0, 1)
            # Y_output = [y_output[index], scores[index]]
            X_params = np.concatenate((X_params, values_resi_mean), axis=0)
            Y_output = [scores[index], df_next_bd.values[index]]
            X.append(X_params)
            Y.append(Y_output)
            metas.append(meta)
            #yield meta, X_params, Y_output
        X = np.array(X)
        #X_argsorts = X[:, :2, :].argsort(axis=0) / 19.
        #yield metas, np.concatenate((X, X_argsorts), axis=1), np.array(Y)
        yield metas, X, np.array(Y)

# def ai_evaluate_pca(df_source_all, date_from_burned, date_from, date_to, symbols_selected, run_round=0):
#     symbols_selected = symbols_selected[:]
#     symbols_selected.sort()
#
#     df_source = df_source_all.loc[(df_source_all.index > date_from_burned) & (df_source_all.index < date_to)]
#     df_closes = df_source[symbols_selected]
#
#     metas, data_X, data_Y = [], [], []
#     for meta, X_params, Y_output in generate_data_pca_x(df_closes, SIZE_WINDOW, N_PAST_VARIABLES, N_COMPONENTS):
#         # for meta, X_params, Y_output in generate_data_tech(df_pivo, date_from, N_PAST_VARIABLES):
#         data_X.append(X_params)
#         data_Y.append(Y_output)
#         metas.append(meta)
#     data_X, data_Y = np.array(data_X), np.array(data_Y)
#     #     print(SIZE_WINDOW, N_PAST_VARIABLES, data_X.shape, df_closes.shape)
#     #     print(df_closes.index[0], df_closes.index[-1])
#
#     return data_X, data_Y, metas
#
#
# def prepare_data_blocks_pca(df_correlations, df_closes, calendar):
#     period_from, period_to = df_correlations.head(1)['date_to'].values[0], df_correlations.tail(1)['date_to'].values[0]
#     days_total_count = df_correlations.shape[0]
#     print(f'Preparing PCA data blocks from {period_from} to {period_to} (total samples count: {days_total_count})')
#     t1 = datetime.datetime.now()
#     # df_closes = df_closes[-common_guaranteed_days:]
#
#     Atrain_X, Atrain_y, Atrain_metas = [], [], []
#     for index, row in df_correlations.iterrows():
#         if index % 20 == 0:  # 100
#             print(index)
#
#         date_from_burned, date_from, date_to, symbols_selected = \
#             row['date_from_burned'], row['date_from'], row['date_to'], row['symbols'].split(',')
#
#         # date_from_burned = calendar.move_by_days(date_from_burned, MOVE_BY_DAYS)
#         # date_from = calendar.move_by_days(date_from, MOVE_BY_DAYS)
#         # date_from_validation = calendar.move_by_days(date_from_validation, MOVE_BY_DAYS)
#         # date_to = calendar.move_by_days(date_from_validation, DAYS_VALIDATION_SIZE)
#
#         train_X, train_y, metas_X = ai_evaluate_pca(df_closes, date_from_burned, date_from, date_to, symbols_selected)
#
#         Atrain_X.append(train_X)
#         Atrain_y.append(train_y)
#         Atrain_metas.extend(metas_X)
#
#     train_X = np.vstack(Atrain_X)
#     train_y = np.vstack(Atrain_y)
#
#     td = datetime.datetime.now() - t1
#     print(f'prepare_data_blocks processed in {td.total_seconds():.3f}s,  '
#           f'train: {Atrain_metas[0][1]} - {Atrain_metas[-1][1]}  ')
#     return train_X, train_y
