import numpy as np
import pandas as pd


def train_test_split_upwelling(dsp_std, window_size, feature_names_w):
    inputs_df = pd.DataFrame(columns=feature_names_w)
    outputs_df = pd.DataFrame(columns=['DSP_out'])
    dsp_cols = [col for col in dsp_std.columns if 'DSP' not in col]
    dsp_std[dsp_cols] = dsp_std[dsp_cols].replace(np.nan, -990.0)
    for i in range(window_size, len(dsp_std)):
        inputs = dsp_std.iloc[i - window_size:i]
        output = dsp_std[['DSP']].iloc[i]
        if inputs.isnull().values.any() or output.isnull().values.any():
            continue
        else:
            new_inputs = inputs.values.reshape(1, -1)
            # Convert the array to a DataFrame
            temp_df = pd.DataFrame(new_inputs, columns=inputs_df.columns)
            inputs_df = pd.concat([inputs_df, temp_df], ignore_index=True)

            temp_df = pd.DataFrame(output.values, columns=outputs_df.columns)
            outputs_df = pd.concat([outputs_df, temp_df], ignore_index=True)

    dsp_input_cols = [col for col in inputs_df.columns if 'DSP' not in col]
    inputs_df[dsp_input_cols] = inputs_df[dsp_input_cols].replace(-990.0, np.nan)
    strings_to_match = dsp_std.columns

    # Concatenate the dataframes
    inputs_df = pd.concat([inputs_df, outputs_df], axis=1)
    test_size = round(inputs_df.shape[0] * 0.2)
    #inputs_df = inputs_df.drop(['DSP_-1week'], axis=1)
    train_inputs = inputs_df.copy().iloc[:-test_size]
    dsp_diff, dsp_min = 0, 0
    test_inputs = inputs_df.copy().iloc[-test_size:]
    for string in strings_to_match:
        matching_columns = [col for col in train_inputs.columns if string in col]
        train_subset = train_inputs.loc[:, matching_columns]
        test_subset = test_inputs.loc[:, matching_columns]
        max = train_subset.melt().value.max()
        min = train_subset.melt().value.min()
        # print(string)
        # print(max)
        # print(min)
        # mean = train_subset.melt().value.mean()
        train_subset = train_subset.applymap(lambda x: (x - min) / (max - min))
        test_subset = test_subset.applymap(lambda x: (x - min) / (max - min))
        if string == 'DSP':
            dsp_diff = max - min
            dsp_min = min
        train_inputs[matching_columns] = train_subset
        test_inputs[matching_columns] = test_subset

    train_outputs = train_inputs[['DSP_out']]
    test_outputs = test_inputs[['DSP_out']]
    train_inputs = train_inputs.drop(['DSP_out'], axis=1)
    test_inputs = test_inputs.drop(['DSP_out'], axis=1)
    train_inputs = train_inputs.replace(np.nan, -990.0)
    test_inputs = test_inputs.replace(np.nan, -990.0)

    print('Train Size: ' + str(train_inputs.shape[0]) + ' Test Size: ' + str(test_inputs.shape[0]))

    return train_inputs, train_outputs, test_inputs, test_outputs, dsp_min, dsp_diff


def train_test_split_upwelling_v3(dsp_std, window_size, df_year, feature_names_w):
    inputs_df = pd.DataFrame(columns=feature_names_w)
    outputs_df = pd.DataFrame(columns=['DSP_out'])
    dsp_cols = [col for col in dsp_std.columns if 'DSP' not in col]
    dsp_std[dsp_cols] = dsp_std[dsp_cols].replace(np.nan, -990.0)
    for i in range(window_size, len(dsp_std)):
        inputs = dsp_std.iloc[i - window_size:i]
        output = dsp_std[['DSP']].iloc[i]
        inputs_year = df_year.iloc[i - window_size:i]
        output_year = df_year.iloc[i]
        if inputs.isnull().values.any() or output.isnull().values.any() or inputs_year['year'].nunique() != 1 or \
                inputs_year['year'].iloc[0] != output_year['year']:
            continue
        else:
            new_inputs = inputs.values.reshape(1, -1)
            # Convert the array to a DataFrame
            temp_df = pd.DataFrame(new_inputs, columns=inputs_df.columns)
            inputs_df = pd.concat([inputs_df, temp_df], ignore_index=True)

            temp_df = pd.DataFrame(output.values, columns=outputs_df.columns)
            outputs_df = pd.concat([outputs_df, temp_df], ignore_index=True)

    dsp_input_cols = [col for col in inputs_df.columns if 'DSP' not in col]
    inputs_df[dsp_input_cols] = inputs_df[dsp_input_cols].replace(-990.0, np.nan)
    strings_to_match = dsp_std.columns

    # Concatenate the dataframes
    inputs_df = pd.concat([inputs_df, outputs_df], axis=1)
    # Features to remove
    #features_to_remove = remove_features(dsp_std, window_size)
    #inputs_df = inputs_df.drop(features_to_remove, axis=1)
    test_size = round(inputs_df.shape[0] * 0.2)
    train_inputs = inputs_df.copy().iloc[:-test_size]
    dsp_diff, dsp_min = 0, 0
    test_inputs = inputs_df.copy().iloc[-test_size:]
    for string in strings_to_match:
        matching_columns = [col for col in train_inputs.columns if string in col]
        train_subset = train_inputs.loc[:, matching_columns]
        test_subset = test_inputs.loc[:, matching_columns]
        max = train_subset.melt().value.max()
        min = train_subset.melt().value.min()
        # mean = train_subset.melt().value.mean()
        train_subset = train_subset.applymap(lambda x: (x - min) / (max - min))
        test_subset = test_subset.applymap(lambda x: (x - min) / (max - min))
        if string == 'DSP':
            dsp_diff = max - min
            dsp_min = min
        train_inputs[matching_columns] = train_subset
        test_inputs[matching_columns] = test_subset

    train_outputs = train_inputs[['DSP_out']]
    test_outputs = test_inputs[['DSP_out']]
    train_inputs = train_inputs.drop(['DSP_out'], axis=1)
    test_inputs = test_inputs.drop(['DSP_out'], axis=1)
    train_inputs = train_inputs.replace(np.nan, -990.0)
    test_inputs = test_inputs.replace(np.nan, -990.0)

    print('Train Size: ' + str(train_inputs.shape[0]) + ' Test Size: ' + str(test_inputs.shape[0]))

    return train_inputs, train_outputs, test_inputs, test_outputs, dsp_min, dsp_diff


def get_features_names(dsp_std, best_window_size):
    feature_names = np.array([])
    for w in reversed(range(1, best_window_size + 1)):
        for c in dsp_std.columns:
            feature_names = np.append(feature_names, c + "_-" + str(w) + "week")

    return feature_names


def remove_features(dsp_std, best_window_size):
    features_to_remove = np.array([])
    for w in range(int(best_window_size / 2) + 1, best_window_size + 1):
        for c in dsp_std.columns:
            if c not in ['average temp', 'total área', 'min temp', 'max temp', 'mean dist', 'max dist', 'temp diff',
                         'min dist']:
                features_to_remove = np.append(features_to_remove, c + "_-" + str(w) + "week")

    return features_to_remove


def convert_class(train_outputs, test_outputs, dsp_diff, dsp_min):
    train_outputs = train_outputs.applymap(lambda x: (x * dsp_diff) + dsp_min)
    test_outputs = test_outputs.applymap(lambda x: (x * dsp_diff) + dsp_min)
    train_outputs.where(train_outputs >= 160, 0, inplace=True)
    train_outputs.where(train_outputs < 160, 1, inplace=True)
    test_outputs.where(test_outputs >= 160, 0, inplace=True)
    test_outputs.where(test_outputs < 160, 1, inplace=True)
    return train_outputs, test_outputs
