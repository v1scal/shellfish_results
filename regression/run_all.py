import pandas as pd
import os
from rf_grid import run_rf
from svr_grid import run_svr
from make_plots import multiple_window_errors

zones = ['l2_Leça Palmeira','l5b_Caparica']
is_rf = False

for z in zones:
    title = z.replace("_", " ").title()
    if is_rf:
        title = title + ' RF Regression'
        alg_type = 'rf'
    else:
        title = title + ' SVR Regression'
        alg_type = 'svr'

    window_errors_dsp = None

    window_errors_meteo = None

    window_errors_hydro = None

    window_errors_all = None

    folder_path = '../data/dsp_meteo'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'imputed_{z}_meteo_dsp_weeks.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_weeks.csv")
        file_name_shortened = file_name_shortened.lstrip("imputed_")
        file_name_shortened += f'_{alg_type}'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week', 'ESTACAO'], axis=1)
        if z in ['l5b_Caparica']:
            df_drop = df_drop.drop(['PR_DUR'], axis=1)
        window_errors_meteo = run_svr(df_drop, file_name_shortened, None, title)
    
    folder_path = '../data/dsp_water'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'imputed_{z}_water_dsp_weeks.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_weeks.csv")
        file_name_shortened = file_name_shortened.lstrip("imputed_")
        file_name_shortened += f'_{alg_type}'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week'], axis=1)
        window_errors_hydro = run_svr(df_drop, file_name_shortened, None, title)

    folder_path = '../data/dsp_meteo_hydro_water'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'imputed_{z}_water_hydro_meteo_dsp_weeks.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_weeks.csv")
        file_name_shortened = file_name_shortened.lstrip("imputed_")
        file_name_shortened += f'_{alg_type}'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week', 'ESTACAO'], axis=1)
        window_errors_all = run_svr(df_drop, file_name_shortened, None, title)

    folder_path = '../data/best_dsp'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_weekly_dsp.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_weekly_dsp.csv")
        file_name_shortened += f'_{alg_type}'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_drop = df_pre.drop(['date', 'week', 'assigned'], axis=1)
        window_errors_dsp = run_svr(df_drop, file_name_shortened, None, title)

    #multiple_window_errors(range(2, 9), window_errors_dsp, window_errors_meteo, window_errors_hydro, window_errors_all,
     #                      f'{z}_{alg_type}', False)
