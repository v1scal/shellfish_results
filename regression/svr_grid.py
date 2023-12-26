import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.svm import SVR
from auxiliary_functions import get_features_names, train_test_split_upwelling, train_test_split_upwelling_v3, remove_features
from make_plots import cm_plot, predictions_plot, window_errors, param_plot, metric_plot, fi_plot, generate_test_metrics, generate_best_params

def run_svr(df, prefix_dsp, df_year, title):
    dsp_std = df
    prefix = prefix_dsp
    if df_year is not None:
        window_sizes = range(2, 6)
    else:
        window_sizes = range(2, 9)
    folds = 5
    best_window_size = None
    window_best_params = None
    window_min_scores = []
    #window_mean_scores = []
    rmse_train = None
    rmse_val = None
    mse_train = None
    mse_val = None
    mae_train = None
    mae_val = None
    r2_train = None
    r2_val = None
    std_rmse_train = None
    std_rmse_val = None
    std_mse_train = None
    std_mse_val = None
    std_mae_train = None
    std_mae_val = None
    std_r2_train = None
    std_r2_val = None

    param_grid = {
        'epsilon':  [0.01, 0.05, 0.1, 0.5, 1],
        'kernel': ['linear', 'rbf', 'poly'],  # Include 'poly' for polynomial kernel
        'C':  [0.1, 0.5, 1, 2, 3, 5, 7, 10],
        'gamma': [0.1, 0.5, 1, 2, 3, 5, 7, 10],
        'degree': [2, 3, 4, 5]  # Degree for the polynomial kernel
    }

    scoring = {'Mean Absolute Error': 'neg_mean_absolute_error', 'Mean Squared Error': 'neg_mean_squared_error', 'R2': 'r2',
               'Root Mean Squared Error': 'neg_root_mean_squared_error'}

    # Define the dimensions of the 3D matrix
    rows = len(param_grid.items())
    columns = len(scoring.items())
    depth = len(window_sizes)

    #  Create an empty 3D matrix with empty lists
    scores = [[[None for _ in range(columns)] for _ in range(rows)] for _ in range(depth)]

    # Get the scoring metric names
    scoring_names = list(scoring.keys())

    for w_id, window_size in enumerate(window_sizes):

        feature_names_w = get_features_names(dsp_std, window_size)

        # Split the data into training and testing sets using a walk-forward approach
        if df_year is not None:
            train_inputs, train_outputs, _, _, dsp_min, dsp_diff = train_test_split_upwelling_v3(dsp_std, window_size,
                                                                                                 df_year,
                                                                                                 feature_names_w)
        else:
            train_inputs, train_outputs, _, _, dsp_min, dsp_diff = train_test_split_upwelling(dsp_std, window_size,
                                                                                              feature_names_w)

        rf = SVR()
        tscv = TimeSeriesSplit(n_splits=folds, gap=window_size - 1)
        gsearch = GridSearchCV(estimator=rf, cv=tscv, param_grid=param_grid, scoring=scoring,
                               refit='Mean Absolute Error', return_train_score=True, n_jobs=-1)

        gsearch.fit(train_inputs, train_outputs.values.ravel())

        print('Best Params for window size: ' + str(window_size))
        print(gsearch.best_params_)
        g_index = gsearch.best_index_


        mse_columns_to_compute = ['mean_test_Mean Squared Error',
                              'mean_train_Mean Squared Error']
        other_columns_to_compute = ['mean_test_Mean Absolute Error',
                              'mean_test_Root Mean Squared Error',
                              'mean_train_Mean Absolute Error',
                              'mean_train_Root Mean Squared Error']

        df_results = pd.DataFrame(gsearch.cv_results_)
        df_results[mse_columns_to_compute] = df_results[mse_columns_to_compute].applymap(
            lambda x: ((-1 * x) * (dsp_diff ** 2)))
        df_results[other_columns_to_compute] = df_results[other_columns_to_compute].applymap(
            lambda x: ((-1 * x) * dsp_diff))

        if not window_min_scores or df_results['mean_test_Mean Absolute Error'][g_index].mean() < min(window_min_scores):
            best_window_size = window_size
            window_best_params = gsearch.best_params_
            rmse_train = df_results['mean_train_Root Mean Squared Error'][g_index].mean()
            rmse_val = df_results['mean_test_Root Mean Squared Error'][g_index].mean()
            mse_train = df_results['mean_train_Mean Squared Error'][g_index].mean()
            mse_val = df_results['mean_test_Mean Squared Error'][g_index].mean()
            mae_train = df_results['mean_train_Mean Absolute Error'][g_index].mean()
            mae_val = df_results['mean_test_Mean Absolute Error'][g_index].mean()
            r2_train = df_results['mean_train_R2'][g_index].mean()
            r2_val = df_results['mean_test_R2'][g_index].mean()
            std_rmse_train = df_results['std_train_Root Mean Squared Error'][g_index].mean()
            std_rmse_val = df_results['std_test_Root Mean Squared Error'][g_index].mean()
            std_mse_train = df_results['std_train_Mean Squared Error'][g_index].mean()
            std_mse_val = df_results['std_test_Mean Squared Error'][g_index].mean()
            std_mae_train = df_results['std_train_Mean Absolute Error'][g_index].mean()
            std_mae_val = df_results['std_test_Mean Absolute Error'][g_index].mean()
            std_r2_train = df_results['std_train_R2'][g_index].mean()
            std_r2_val = df_results['std_test_R2'][g_index].mean()

        window_min_scores.append(df_results['mean_test_Mean Absolute Error'][g_index].mean())
        #window_mean_scores.append(df_results['mean_test_Mean Absolute Error'].mean())

        results = ['mean_test_Mean Absolute Error',
                   'mean_test_Mean Squared Error',
                   'mean_test_R2',
                   'mean_test_Root Mean Squared Error']

        for p_id, (param_name, param_range) in enumerate(param_grid.items()):
            #df_results = df_results.replace(np.nan, 40)
            grouped_df = df_results.groupby(f'param_{param_name}')[results].mean()
            for s_id, (scoring_name, scoring_code) in enumerate(scoring.items()):
                scores[w_id][p_id][s_id] = grouped_df[f'mean_test_{scoring_name}'].values

    feature_names = get_features_names(dsp_std, best_window_size)

    param_plot(scoring_names, param_grid, window_sizes, scores, prefix)

    metric_plot(scoring_names, param_grid, window_sizes, scores, prefix)

    window_errors(window_sizes, window_min_scores, prefix, True)

    if df_year is not None:
        train_inputs, train_outputs, test_inputs, test_outputs, dsp_min, dsp_diff = train_test_split_upwelling_v3(
            dsp_std, best_window_size, df_year, feature_names)
    else:
        train_inputs, train_outputs, test_inputs, test_outputs, dsp_min, dsp_diff = \
            train_test_split_upwelling(dsp_std, best_window_size, feature_names)

    # Train the final model using the best hyperparameters found across all folds
    best_rf = SVR(**window_best_params)
    best_rf.fit(train_inputs, train_outputs.values.ravel())
    #importances = best_rf.feature_importances_

    """
    if df_year is not None:
        feature_filter = remove_features(dsp_std, best_window_size)
        filtered_features = [string for string in feature_names if string not in feature_filter]
        fi_plot(np.array(filtered_features), importances, prefix)
    else:
        fi_plot(np.array(feature_names), importances, prefix)
    """
    # Make predictions on the test set
    y_pred = best_rf.predict(test_inputs)
    y_real = test_outputs
    print(dsp_diff)
    # Convert y_real to a numpy array
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    # Perform element-wise multiplication and addition

    y_real = y_real * dsp_diff + dsp_min
    y_pred = y_pred * dsp_diff + dsp_min

    generate_best_params(window_best_params, prefix)
    generate_test_metrics(y_real, y_pred, prefix, mae_val, mae_train, rmse_val, rmse_train, mse_val, mse_train, r2_val, r2_train, dsp_diff, dsp_min, std_mae_train, std_mae_val, std_r2_val, std_r2_train, std_mse_val, std_mse_train, std_rmse_val, std_rmse_train)



    predictions_plot(y_real, y_pred, prefix, True)
    cm_plot(y_real, y_pred, prefix, True, title)

    return window_min_scores
