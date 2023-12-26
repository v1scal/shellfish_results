import pandas as pd
import os
from rf_grid import run_rf
from svr_grid import run_svr
from make_plots import multiple_upwelling_window_errors

zones = ['l1_Carreço','RIAV1_Triangulo']
is_rf = True

for z in zones:
    title = z.replace("_", " ").title()
    if is_rf:
        title = title + ' RF Regression'
        alg_type = 'rf'
    else:
        title = title + ' SVR Regression'
        alg_type = 'svr'

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

    y_real_meteo_hydro = None
    y_pred_meteo_hydro = None
    window_errors_meteo_hydro = None
    proba_meteo_hydro = None

    y_real_meteo_hydro_upwelling = None
    y_pred_meteo_hydro_upwelling = None
    window_errors_meteo_hydro_upwelling = None
    proba_meteo_hydro_upwelling = None

    y_real_meteo_upwelling = None
    y_pred_meteo_upwelling = None
    window_errors_meteo_upwelling = None
    proba_meteo_upwelling = None

    y_real_dsp_upwelling = None
    y_pred_dsp_upwelling = None
    window_errors_dsp_upwelling = None
    proba_dsp_upwelling = None

    y_real_hydro_upwelling = None
    y_pred_hydro_upwelling = None
    window_errors_hydro_upwelling = None
    proba_hydro_upwelling = None

    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp.csv")
        file_name_shortened += f'_dsp_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance'], axis=1)
        window_errors_dsp = run_rf(df_drop, file_name_shortened, df_year, title)
    '''
    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp_hydro.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_hydro.csv")
        file_name_shortened += f'_hydro_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance'], axis=1)
        window_errors_hydro = run_svr(df_drop, file_name_shortened, df_year, title)

    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp_hydro_upwelling.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_hydro_upwelling.csv")
        file_name_shortened += f'_hydro_upwelling_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance'], axis=1)
        window_errors_hydro_upwelling = run_svr(
            df_drop,
            file_name_shortened,
            df_year,
            title)
    '''
    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp_meteo.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_meteo.csv")
        file_name_shortened += f'_meteo_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance', 'year'], axis=1)
        if z in ['l5b_Caparica']:
            df_drop = df_drop.drop(['PR_DUR'], axis=1)
        window_errors_meteo = run_rf(df_drop, file_name_shortened, df_year, title)
    '''
    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp_meteo_hydro.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_meteo_hydro.csv")
        file_name_shortened += f'_meteo_hydro_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance', 'year'], axis=1)
        window_errors_meteo_hydro = run_svr(df_drop, file_name_shortened, df_year, title)

    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp_meteo_hydro_upwelling.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_meteo_hydro_upwelling.csv")
        file_name_shortened += f'_meteo_hydro_upwelling_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance', 'year'], axis=1)
        window_errors_meteo_hydro_upwelling = run_svr(
            df_drop, file_name_shortened, df_year, title)
    '''
    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp_meteo_upwelling.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_meteo_upwelling.csv")
        file_name_shortened += f'_meteo_upwelling_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance', 'year'], axis=1)
        if z in ['l5b_Caparica']:
            df_drop = df_drop.drop(['PR_DUR'], axis=1)
        window_errors_meteo_upwelling = run_rf(
            df_drop,
            file_name_shortened,
            df_year,
            title)

    folder_path = '../data/upwelling_datasets/dsp_meteo_hydro_upwelling'
    for file_name in os.listdir(folder_path):
        if file_name not in [f'{z}_dsp_upwelling.csv']:
            continue
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        file_name_shortened = file_name.rstrip("_dsp_upwelling.csv")
        file_name_shortened += f'_dsp_upwelling_{alg_type}_up'
        print(file_name_shortened)
        df_pre = pd.read_csv(file_path)
        df_year = df_pre['instance'].str[:4].rename('year').to_frame()
        df_drop = df_pre.drop(['date', 'ESTACAO', 'instance'], axis=1)
        window_errors_dsp_upwelling = run_rf(
            df_drop,
            file_name_shortened,
            df_year, title)

    multiple_upwelling_window_errors(range(2, 6), window_errors_dsp, window_errors_meteo,
                                     window_errors_hydro, window_errors_meteo_hydro,
                                     window_errors_dsp_upwelling, window_errors_meteo_upwelling,
                                     window_errors_hydro_upwelling, window_errors_meteo_hydro_upwelling,
                                     f'{z}_upwelling_{alg_type}', False)
