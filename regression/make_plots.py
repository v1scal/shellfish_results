import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_curve, roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score
from prettytable import PrettyTable
import plotly.figure_factory as ff


def multiple_window_errors(window_sizes, window_min_scores_dsp, window_min_scores_meteo, window_min_scores_hydro,
                           window_min_scores_all, prefix, regression):
    plt.plot(window_sizes, window_min_scores_dsp, label="DSP")
    plt.plot(window_sizes, window_min_scores_meteo, label="Meteo")

    #plt.plot(window_sizes, window_min_scores_hydro, label="Hydro")
    #plt.plot(window_sizes, window_min_scores_all, label="Meteo and Hydro")

    # plt.plot(window_sizes, window_mean_scores, label="Mean MAE")
    plt.legend(fontsize='small', loc='upper right')
    plt.title('Window Size Optimization')
    plt.xlabel('Window Size')
    if regression:
        plt.ylabel('Mean Absolute Error')
    else:
        plt.ylabel('MAE')
    plt.savefig(f'./best_plots/{prefix}_all_window_regression.png', bbox_inches='tight')
    plt.show()
    plt.close()

def multiple_upwelling_window_errors(window_sizes, window_min_scores_dsp, window_min_scores_meteo,
                                     window_min_scores_hydro, window_min_scores_meteo_hydro,
                                     window_min_scores_dsp_upwelling, window_min_scores_meteo_upwelling,
                                     window_min_scores_hydro_upwelling, window_min_scores_meteo_hydro_upwelling, prefix,
                                     regression):
    plt.plot(window_sizes, window_min_scores_dsp, label="DSP")
    plt.plot(window_sizes, window_min_scores_meteo, label="Meteo")
    '''
    plt.plot(window_sizes, window_min_scores_hydro, label="Hydro")
    plt.plot(window_sizes, window_min_scores_meteo_hydro, label="Meteo and Hydro")
    '''
    plt.plot(window_sizes, window_min_scores_dsp_upwelling, label="DSP and Upwelling")
    plt.plot(window_sizes, window_min_scores_meteo_upwelling, label="Meteo and Upwelling")
    '''
    plt.plot(window_sizes, window_min_scores_hydro_upwelling, label="Hydro and Upwelling")
    plt.plot(window_sizes, window_min_scores_meteo_hydro_upwelling, label="Meteo, Hydro and Upwelling")
    '''
    # plt.plot(window_sizes, window_mean_scores, label="Mean MAE")
    plt.legend()
    plt.title('Window Size Optimization')
    plt.xlabel('Window Size')
    if regression:
        plt.ylabel('Mean Absolute Error')
    else:
        plt.ylabel('MAE')
    plt.savefig(f'./best_plots/{prefix}_all_upwelling_window.png', bbox_inches='tight')
    plt.show()
    plt.close()

def roc_auc_plot(y_real, y_pred, prefix):
    fpr, tpr, _ = roc_curve(y_real, y_pred)
    auc = roc_auc_score(y_real, y_pred)

    plt.plot(fpr, tpr, label='AUC=' + str(auc))
    # plt.plot(y_pred, label='Predicted Values')
    plt.title('ROC and AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'./best_plots/{prefix}_roc_auc_score_regression.png')
    plt.show()
    plt.close()


def cm_plot(y_real, y_pred, prefix, regression, title):
    y_real_dual = pd.DataFrame(y_real)
    y_pred_dual = pd.DataFrame(y_pred)
    if regression:
        y_real_dual.where(y_real_dual >= 160, 0, inplace=True)
        y_real_dual.where(y_real_dual < 160, 1, inplace=True)

        y_pred_dual.where(y_pred_dual >= 160, 0, inplace=True)
        y_pred_dual.where(y_pred_dual < 160, 1, inplace=True)
    cm = confusion_matrix(y_real_dual, y_pred_dual)
    if cm.shape[0] > 1 and cm.shape[1] > 1:
        false_positives = cm[0, 1]
        true_negatives = cm[0, 0]
        false_positive_rate = false_positives / (false_positives + true_negatives)
        true_positives = cm[1, 1]
        false_negatives = cm[1, 0]
        false_negative_rate = false_negatives / (true_positives + false_negatives)
        generate_fp_fn(false_positives, false_negatives, false_positive_rate, false_negative_rate, prefix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Contamined', 'Contamined'])
    disp.plot(cmap='Blues', colorbar = False)
    plt.title(title)
    plt.savefig(f'./best_plots/{prefix}_confusionmatrix_regression.png')
    plt.show()
    plt.close()


def predictions_plot(y_real, y_pred, prefix, regression):
    plt.plot(y_real, label='Observed Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.xticks(ticks=range(0, len(y_real), 1), labels=range(0, len(y_real), 1))
    plt.tick_params(axis='x', which='major', pad=8)
    if regression:
        plt.axhline(y=160, color='r', linestyle='--')
    plt.title('Observed vs Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('DSP Concentration')
    plt.legend()
    plt.savefig(f'./best_plots/{prefix}_predictions_regression.png', bbox_inches='tight')
    plt.show()
    plt.close()


def window_errors(window_sizes, window_min_scores, prefix, regression):
    plt.plot(window_sizes, window_min_scores, label="Min MAE")
    # plt.plot(window_sizes, window_mean_scores, label="Mean MAE")
    plt.legend()
    plt.title('Window Size Optimization')
    plt.xlabel('Window Size')
    if regression:
        plt.ylabel('Mean Absolute Error')
    else:
        plt.ylabel('F1')
    plt.savefig(f'./best_plots/{prefix}_window_regression.png', bbox_inches='tight')
    plt.show()
    plt.close()


def param_plot(scoring_names, param_grid, window_sizes, scores, prefix):
    # Loop over the parameter names
    for i, (param_name, param_range) in enumerate(param_grid.items()):
        fig, axes = plt.subplots(1, len(scoring_names), figsize=(12, 4))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # Loop over the scoring metrics
        for f, scoring_name in enumerate(scoring_names):

            # Create a separate plot for each scoring metric
            ax = axes[f]
            for w_id, w in enumerate(window_sizes):
                ax.plot(np.arange(len(param_range)), scores[w_id][i][f], '-o', label=("w_" + str(w)))

            ax.set_xticks(np.arange(len(param_range)))
            ax.set_xticklabels(param_range, fontsize='small')
            ax.set_xlabel(param_name)
            ax.set_ylabel(scoring_name)
            ax.legend()

        # Set the overall title and adjust the spacing between the subplots
        fig.suptitle(f"{param_name}", fontsize=16)
        fig.tight_layout()
        plt.savefig(f'./best_plots/{prefix}_{param_name}_regression.png')

        plt.show()
        plt.close()


def metric_plot(scoring_names, param_grid, window_sizes, scores, prefix):
    # Loop over the parameter names
    for f, scoring_name in enumerate(scoring_names):
        fig, axes = plt.subplots(1, len(param_grid.items()), figsize=(12, 4))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        # Loop over the scoring metrics
        for i, (param_name, param_range) in enumerate(param_grid.items()):

            # Create a separate plot for each scoring metric
            ax = axes[i]
            for w_id, w in enumerate(window_sizes):
                ax.plot(param_range, scores[w_id][i][f], '-o', label=("w_" + str(w)))

            ax.set_xticks(param_range)
            ax.set_xlabel(param_name)
            ax.set_ylabel(scoring_name)
            ax.legend()

        # Set the overall title and adjust the spacing between the subplots
        fig.suptitle(f"{scoring_name}", fontsize=16)
        fig.tight_layout()
        plt.savefig(f'./best_plots/{prefix}_{scoring_name}_regression.png')

        plt.show()
        plt.close()


def fi_plot(feature_names, importances, prefix):
    # importances = importances.reshape((-1, 1))
    # Concatenate the arrays into a 2D array
    # importances = importances.astype(float)
    fi = np.concatenate((feature_names.reshape(-1, 1), importances.reshape(-1, 1)), axis=1)
    # fi = np.concatenate((feature_names, importances), axis=0)
    # print(fi)
    df = pd.DataFrame(fi, columns=['a', 'b'])

    df[['b']] = df[['b']].applymap(lambda x: format(float(x), '.8f'))
    print(df)
    fi = df.values

    # convert the second column of `fi` to a float array
    fi[:, 1] = fi[:, 1].astype(float)

    # sort the array by the second column (importances) in descending order
    sorted_indices = np.argsort(fi[:, 1])[::-1]

    # select the first 10 rows
    top_10_indices = sorted_indices[:10]

    # get the top 10 entries from the sorted array
    top_10_entries = fi[top_10_indices]

    # display the top 10 entries
    print(top_10_entries)

    plt.bar(top_10_entries[:, 0], top_10_entries[:, 1])
    plt.title('Feature Importance on Final Model')
    plt.xlabel('Feature Name')
    plt.ylabel('Mean Decrease in Impurity')
    plt.tick_params(axis='x', which='major', pad=8, labelrotation=45)
    plt.savefig(f'./best_plots/{prefix}_fi_regression.png', bbox_inches='tight')
    plt.show()
    plt.close()


def generate_test_metrics(y_real, y_pred, prefix, mae_val, mae_train, rmse_val, rmse_train, mse_val, mse_train, r2_val,
                          r2_train, dsp_diff, dsp_min, std_mae_train, std_mae_val, std_r2_val, std_r2_train,
                          std_mse_val, std_mse_train, std_rmse_val, std_rmse_train):
    mse = mean_squared_error(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred, squared=False)
    mae = mean_absolute_error(y_real, y_pred)
    # mse = mse * (dsp_diff ** 2)
    # rmse = rmse * dsp_diff
    # mae = mae * dsp_diff
    r2 = r2_score(y_real, y_pred)
    table = PrettyTable()
    table.field_names = ["Test Mean Squared Error", "Test Root Mean Squared Error", "Test Mean Absolute Error",
                         "Test R2"]
    table.add_row([mse, rmse, mae, r2])
    print(table)
    '''
    df = pd.DataFrame(
        [['Mean Square Error', mse_train, std_mse_train, mse_val, std_mse_val, mse], ['Root Mean Square Error', rmse_train, std_rmse_train, rmse_val, std_rmse_val, rmse],
         ['Mean Absolute Error', mae_train, std_mae_train, mae_val, std_mae_val, mae], ['R2', r2_train, std_r2_train, r2_val, std_r2_val, r2]],
        columns=['Metric', 'Training Error', 'Training STD', 'Validation Error', 'Validation STD', 'Testing Error'])
    '''
    df = pd.DataFrame(
        [['Mean Square Error', mse_train, mse_val, mse], ['Root Mean Square Error', rmse_train, rmse_val, rmse],
         ['Mean Absolute Error', mae_train, mae_val, mae], ['R2', r2_train, r2_val, r2]],
        columns=['Metric', 'Training Error', 'Validation Error', 'Testing Error'])
    fig = ff.create_table(df)
    fig.update_layout(
        autosize=True
    )
    fig.write_image(f"./best_plots/{prefix}_table_metrics_regression.png", engine='kaleido')
    plt.show()
    plt.close()


def generate_test_metrics_class(y_real, y_pred, prefix, dsp_diff, dsp_min, accuracy_train, accuracy_val, precision_train,
                                precision_val, f1_train, f1_val, recall_train, recall_val, roc_auc_train, roc_auc_val, std_accuracy_train,
                                std_accuracy_val, std_precision_train, std_precision_val, std_f1_train, std_f1_val,
                                std_recall_train, std_recall_val):
    acc = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    roc_auc = roc_auc_score(y_real, y_pred)
    table = PrettyTable()
    table.field_names = ["Test Accuracy", "Test Root Precision", "Test F1",
                         "Test Recall"]
    table.add_row([acc, precision, f1, recall])
    print(table)

    df = pd.DataFrame(
        [['Accuracy', accuracy_train, accuracy_val, acc], ['Precision', precision_train, precision_val, precision],
         ['F1', f1_train, f1_val, f1], ['Recall', recall_train, recall_val, recall]],
        columns=['Metric', 'Training Error', 'Validation Error', 'Testing Error'])
    fig = ff.create_table(df)
    fig.update_layout(
        autosize=True
    )
    fig.write_image(f"./best_plots/{prefix}_table_metrics_regression.png", engine='kaleido')
    plt.show()
    plt.close()


def generate_best_params(window_best_params, prefix):
    df = pd.DataFrame([window_best_params.values()], columns=window_best_params.keys())
    fig = ff.create_table(df)
    fig.update_layout(
        autosize=True
    )
    fig.write_image(f"./best_plots/{prefix}_table_params_regression.png", engine='kaleido')
    plt.show()
    plt.close()


def generate_fp_fn(fp, fn, fpr, fnr, prefix):
    df = pd.DataFrame([[int(fp), fpr, int(fn), fnr]],
                      columns=['False Positive', 'False Positive Rate', 'False Negative', 'False Negative Rate'])
    fig = ff.create_table(df)
    fig.update_layout(
        autosize=True
    )
    fig.write_image(f"./best_plots/{prefix}_table_fp_fn_regression.png", engine='kaleido')
    plt.show()
    plt.close()
