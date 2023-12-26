import pandas as pd
import os
from rf_classification import run_rf_class
from svm_classification import run_svm_class
from make_plots import multiple_pr_re_plot, multiple_roc_auc_plot, multiple_window_errors

zones = ['RIAV1_Triangulo']

for z in zones:
    title = z.replace("_", " ").title()
    title = title + ' SVM Classification'

    y_real_dsp = None
    y_pred_dsp = None
    window_errors_dsp = None
    proba_dsp = None

    y_real_meteo = None
    y_pred_meteo = None
    window_errors_meteo = None
    proba_meteo = None

    y_real_hydro = None
    y_pred_hydro = None
    window_errors_hydro = None
    proba_hydro = None

    y_real_all = None
    y_pred_all = None
    window_errors_all = None
    proba_all = None

    folder_path = '../data/best_dsp'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_weekly_dsp.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_weekly_dsp.csv")
        file_name_shortened += '_svm'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week', 'assigned'], axis=1)

        window_errors_dsp, y_real_dsp, y_pred_dsp, proba_dsp = run_svm_class(df_drop, file_name_shortened, None, None, title)

    folder_path = '../data/dsp_meteo'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'imputed_{z}_meteo_dsp_weeks.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_weeks.csv")
        file_name_shortened = file_name_shortened.lstrip("imputed_")
        file_name_shortened += '_svm'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week', 'ESTACAO'], axis=1)
        if z in ['l5b_Caparica']:
            df_drop = df_drop.drop(['PR_DUR'], axis=1)

        window_errors_meteo, y_real_meteo, y_pred_meteo, proba_meteo = run_svm_class(df_drop, file_name_shortened, None, None, title)

    folder_path = '../data/dsp_water'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'imputed_{z}_water_dsp_weeks.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_weeks.csv")
        file_name_shortened = file_name_shortened.lstrip("imputed_")
        file_name_shortened += '_svm'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week'], axis=1)

        window_errors_hydro, y_real_hydro, y_pred_hydro, proba_hydro = run_svm_class(df_drop, file_name_shortened, None, None, title)

    folder_path = '../data/dsp_meteo_hydro_water'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'imputed_{z}_water_hydro_meteo_dsp_weeks.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_weeks.csv")
        file_name_shortened = file_name_shortened.lstrip("imputed_")
        file_name_shortened += '_svm'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week', 'ESTACAO'], axis=1)

        window_errors_all, y_real_all, y_pred_all, proba_all = run_svm_class(df_drop, file_name_shortened, None, None, title)
    '''
    multiple_window_errors(range(2, 9), window_errors_dsp, window_errors_meteo, window_errors_hydro, window_errors_all,
                           f'{z}_svm', False)
    multiple_roc_auc_plot(y_real_dsp, y_pred_dsp, y_real_meteo, y_pred_meteo, y_real_hydro, y_pred_hydro, y_real_all,
                          y_pred_all, f'{z}_svm', proba_dsp, proba_meteo, proba_hydro, proba_all)
    multiple_pr_re_plot(y_real_dsp, y_pred_dsp, y_real_meteo, y_pred_meteo, y_real_hydro, y_pred_hydro, y_real_all,
                        y_pred_all, f'{z}_svm', proba_dsp, proba_meteo, proba_hydro, proba_all)
    '''