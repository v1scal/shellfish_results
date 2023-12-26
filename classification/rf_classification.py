from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from auxiliary_functions import get_features_names, train_test_split_upwelling, \
    train_test_split_upwelling_v3, remove_features, convert_class, train_test_split_mix
from make_plots import cm_plot, predictions_plot, window_errors, param_plot, metric_plot, fi_plot, \
    generate_test_metrics_class, generate_best_params, roc_auc_plot, pr_re_plot, class_predictions_plot, window_errors_2
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score


def run_rf_class(df, prefix_dsp, df_year, df_assigned, title):
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
    window_mean_scores = []
    accuracy_train = None
    accuracy_val = None
    precision_train = None
    precision_val = None
    f1_train = None
    f1_val = None
    recall_train = None
    recall_val = None
    roc_auc_train = None
    roc_auc_val = None
    gmean_train = None
    gmean_val = None
    std_accuracy_train = None
    std_accuracy_val = None
    std_precision_train = None
    std_precision_val = None
    std_f1_train = None
    std_f1_val = None
    std_recall_train = None
    std_recall_val = None

    param_grid = {'n_estimators': [100, 200, 300, 400],
                  'max_depth': [5, 10, 12, 15, 20],
                  'min_samples_split': [4, 5, 6, 7, 8],
                  'min_samples_leaf': [4, 6, 8, 10, 12]}

    gm_scorer = make_scorer(geometric_mean_score, greater_is_better=True, average='macro')
    scoring = {'Balanced Accuracy': 'balanced_accuracy', 'Average Precision': 'average_precision', 'Recall': 'recall',
               'F1 Macro': 'f1_macro', 'GMean': gm_scorer}

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
            train_outputs, _ = convert_class(train_outputs, _, dsp_diff, dsp_min)
        elif df_assigned is not None:
            train_inputs, train_outputs, _, _, dsp_min, dsp_diff = train_test_split_mix(dsp_std, window_size,
                                                                                        feature_names_w, df_assigned)
            train_outputs, _ = convert_class(train_outputs, _, dsp_diff, dsp_min)
        else:
            train_inputs, train_outputs, _, _, dsp_min, dsp_diff = train_test_split_upwelling(dsp_std, window_size,
                                                                                              feature_names_w)
            train_outputs, _ = convert_class(train_outputs, _, dsp_diff, dsp_min)

        rf = RandomForestClassifier(n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=folds, gap=window_size - 1)
        gsearch = GridSearchCV(estimator=rf, cv=tscv, param_grid=param_grid, scoring=scoring,
                               refit='F1 Macro', return_train_score=True, n_jobs=-1)

        gsearch.fit(train_inputs, train_outputs.values.ravel())

        print('Best Params for window size: ' + str(window_size))
        print(gsearch.best_params_)
        g_index = gsearch.best_index_

        df_results = pd.DataFrame(gsearch.cv_results_)

        if not window_min_scores or df_results['mean_test_F1 Macro'][g_index].mean() > max(window_min_scores):
            best_window_size = window_size
            window_best_params = gsearch.best_params_
            accuracy_train = df_results['mean_train_Balanced Accuracy'][g_index].mean()
            accuracy_val = df_results['mean_test_Balanced Accuracy'][g_index].mean()
            precision_train = df_results['mean_train_Average Precision'][g_index].mean()
            precision_val = df_results['mean_test_Average Precision'][g_index].mean()
            f1_train = df_results['mean_train_F1 Macro'][g_index].mean()
            f1_val = df_results['mean_test_F1 Macro'][g_index].mean()
            recall_train = df_results['mean_train_Recall'][g_index].mean()
            recall_val = df_results['mean_test_Recall'][g_index].mean()
            # roc_auc_train = df_results['mean_train_ROC_AUC'][g_index].mean()
            # roc_auc_val = df_results['mean_test_ROC_AUC'][g_index].mean()
            gmean_train = df_results['mean_train_GMean'][g_index].mean()
            gmean_val = df_results['mean_test_GMean'][g_index].mean()

        window_min_scores.append(df_results['mean_test_F1 Macro'][g_index].mean())
        window_mean_scores.append(df_results['mean_test_F1 Macro'].mean())

        results = ['mean_test_Balanced Accuracy',
                   'mean_test_Average Precision',
                   'mean_test_Recall',
                   'mean_test_F1 Macro',
                   'mean_test_GMean'
                   ]

        for p_id, (param_name, param_range) in enumerate(param_grid.items()):
            # df_results = df_results.replace(np.nan, 'All')
            grouped_df = df_results.groupby(f'param_{param_name}')[results].mean()
            for s_id, (scoring_name, scoring_code) in enumerate(scoring.items()):
                scores[w_id][p_id][s_id] = grouped_df[f'mean_test_{scoring_name}'].values

    feature_names = get_features_names(dsp_std, best_window_size)

    param_plot(scoring_names, param_grid, window_sizes, scores, prefix)

    metric_plot(scoring_names, param_grid, window_sizes, scores, prefix)

    window_errors(window_sizes, window_min_scores, prefix, False)
    window_errors_2(window_sizes, window_min_scores, window_mean_scores, prefix, False)

    if df_year is not None:
        train_inputs, train_outputs, test_inputs, test_outputs, dsp_min, dsp_diff = train_test_split_upwelling_v3(
            dsp_std, best_window_size, df_year, feature_names)
        true_output = test_outputs
        train_outputs, test_outputs = convert_class(train_outputs, test_outputs, dsp_diff, dsp_min)
    elif df_assigned is not None:
        train_inputs, train_outputs, test_inputs, test_outputs, dsp_min, dsp_diff = train_test_split_mix(dsp_std,
                                                                                                         best_window_size,
                                                                                                         feature_names,
                                                                                                         df_assigned)
        true_output = test_outputs
        train_outputs, test_outputs = convert_class(train_outputs, test_outputs, dsp_diff, dsp_min)
    else:
        train_inputs, train_outputs, test_inputs, test_outputs, dsp_min, dsp_diff = \
            train_test_split_upwelling(dsp_std, best_window_size, feature_names)
        true_output = test_outputs
        train_outputs, test_outputs = convert_class(train_outputs, test_outputs, dsp_diff, dsp_min)

    # Train the final model using the best hyperparameters found across all folds
    best_rf = RandomForestClassifier(**window_best_params, n_jobs=-1)
    best_rf.fit(train_inputs, train_outputs.values.ravel())
    importances = best_rf.feature_importances_

    proba = best_rf.predict_proba(test_inputs)[:, 1]

    if df_year is not None:
        # feature_filter = remove_features(dsp_std, best_window_size)
        # filtered_features = [string for string in feature_names if string not in feature_filter]
        fi_plot(np.array(feature_names), importances, prefix)
    else:
        fi_plot(np.array(feature_names), importances, prefix)

    # Make predictions on the test set
    y_pred = best_rf.predict(test_inputs)
    y_real = test_outputs
    print(dsp_diff)
    # Convert y_real to a numpy array
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    # Resetting the index
    true_output = true_output.reset_index(drop=True)
    true_output['DSP_out'] = true_output['DSP_out'].apply(lambda x: x * dsp_diff + dsp_min)
    real_output = true_output.copy()

    # A is real output, B is predicted, C is misclassifications, D is class changes, E is misclassifications during
    # class changes, F is correct classifications during class change and G is misclassifications not during state
    # changes

    real_output['A'] = y_real
    real_output['B'] = y_pred
    real_output['C'] = real_output['A'] - real_output['B']
    real_output['C'] = real_output['C'].apply(lambda x: abs(x))
    other_real_output = real_output.copy()
    real_output = real_output.drop(['A', 'B'], axis=1)
    real_output_index = real_output.loc[real_output['C'] != 0].index
    real_output = real_output[real_output['C'] != 0]

    # Create column D with initial value 0
    other_real_output['D'] = 0

    # Iterate through the rows and set D to 1 if A changed from the previous row
    for i in range(1, len(other_real_output)):
        if other_real_output.at[i, 'A'] != other_real_output.at[i - 1, 'A']:
            other_real_output.at[i, 'D'] = 1

    other_real_output['E'] = ((other_real_output['D'] == 1) & (other_real_output['C'] == 1)).astype(int)

    e_real_output = other_real_output.copy()
    e_real_output_index = e_real_output.loc[e_real_output['E'] != 0].index
    e_real_output = e_real_output[e_real_output['E'] != 0]

    other_real_output['F'] = ((other_real_output['D'] == 1) & (other_real_output['C'] == 0)).astype(int)

    f_real_output = other_real_output.copy()
    f_real_output_index = f_real_output.loc[f_real_output['F'] != 0].index
    f_real_output = f_real_output[f_real_output['F'] != 0]

    other_real_output['G'] = ((other_real_output['D'] == 0) & (other_real_output['C'] == 1)).astype(int)

    g_real_output = other_real_output.copy()
    g_real_output_index = g_real_output.loc[g_real_output['G'] != 0].index
    g_real_output = g_real_output[g_real_output['G'] != 0]

    # Plots to do just C, just E, E + F, E + G
    print(real_output)
    # Perform element-wise multiplication and addition

    generate_best_params(window_best_params, prefix)
    generate_test_metrics_class(y_real, y_pred, prefix, dsp_diff, dsp_min, accuracy_train, accuracy_val,
                                precision_train,
                                precision_val, f1_train, f1_val, recall_train, recall_val, roc_auc_train, roc_auc_val,
                                gmean_train, gmean_val, std_accuracy_train,
                                std_accuracy_val, std_precision_train, std_precision_val, std_f1_train, std_f1_val,
                                std_recall_train, std_recall_val, proba)
    roc_auc_plot(y_real, y_pred, prefix, proba)
    pr_re_plot(y_real, y_pred, prefix, proba)

    class_predictions_plot(true_output['DSP_out'], real_output_index, real_output['DSP_out'], e_real_output_index,
                           e_real_output['DSP_out'], f_real_output_index, f_real_output['DSP_out'], g_real_output_index,
                           g_real_output['DSP_out'], prefix, False)
    cm_plot(y_real, y_pred, prefix, False, title)

    return window_min_scores, y_real, y_pred, proba
