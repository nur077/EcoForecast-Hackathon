import argparse
import os
import glob
import numpy as np
import pandas as pd

def load_data(file_path):
    # TODO: Load data from CSV file
    directory_path = file_path
    non_green_energy = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B14', 'B17', 'B20']
    green_energy = ['B01', 'B09', 'B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18', 'B19']
    prefixes = ['gen_SP', 'gen_UK', 'gen_DE', 'gen_DK', 'gen_HU', 'gen_SE', 'gen_IT', 'gen_PO', 'gen_NE']

    new_load_columns = ['SP_Load', 'UK_Load', 'DE_Load', 'DK_Load', 'HU_Load', 'SE_Load', 'IT_Load', 'PO_Load', 'NE_Load']
    load_columns = ['load_SP.csv', 'load_UK.csv', 'load_DE.csv', 'load_DK.csv', 'load_HU.csv', 'load_SE.csv', 'load_IT.csv', 'load_PO.csv', 'load_NE.csv']

    sp = ['SP', 'UK', 'DE', 'DK', 'HU', 'SE', 'IT', 'PO', 'NL']
    sp_green = ['green_energy_SP', 'green_energy_UK', 'green_energy_DE', 'green_energy_DK','green_energy_HU', 'green_energy_SE', 'green_energy_IT', 'green_energy_PO','green_energy_NE']

    dfs, lst = [], []
    lst_new_dfs = []

    # converting 15, 30 min intervals into 1 hour interval for gen_*.csv
    def new_interval(df):
        df['StartTime'] = pd.to_datetime(df['StartTime'].str[:-1], format='%Y-%m-%dT%H:%M:%S%z')
        df['EndTime'] = pd.to_datetime(df['EndTime'].str[:-1], format='%Y-%m-%dT%H:%M:%S%z')
        df['Hour'] = df['StartTime'].dt.floor('H')
        result = df.groupby(['Hour', 'AreaID', 'UnitName', 'PsrType']).agg({'quantity': 'sum'}).reset_index()
        return result

    # converting 15, 30 min intervals into 1 hour interval for load_*.csv
    def new_interval_load(df):
        df['StartTime'] = pd.to_datetime(df['StartTime'].str[:-1], format='%Y-%m-%dT%H:%M:%S%z')
        df['EndTime'] = pd.to_datetime(df['EndTime'].str[:-1], format='%Y-%m-%dT%H:%M:%S%z')
        df['Hour'] = df['StartTime'].dt.floor('H')
        result = df.groupby(['Hour', 'AreaID', 'UnitName']).agg({'Load': 'sum'}).reset_index()
        return result

    # concatinating all gen_* type csv files, excluding non-green energy
    for prefix in prefixes:
        search = f'{prefix}*.csv'
        files = glob.glob(os.path.join(directory_path, search))

        for file_path in files:
            df1 = pd.read_csv(file_path)
            df = new_interval(df1)
            dfs.append(df)

        current_prefix_df = pd.concat(dfs, ignore_index=True)

        pivot_df = current_prefix_df.pivot(index='Hour', columns='PsrType', values='quantity').fillna(0).reset_index()
        valid_columns = [elem for elem in pivot_df.columns if elem[-3:] in green_energy]
        result_df = pivot_df[valid_columns]
        clmn = f'green_energy_{prefix}'
        result_df = result_df.copy()
        result_df[clmn] = result_df.sum(axis=1)
        lst_new_dfs.append(result_df[clmn])
        dfs = []

    df = pd.concat(lst_new_dfs, axis=1)
    df.columns = df.columns.str.replace('green_energy_gen_', 'green_energy_')

    updated_new_df = df.copy()

    # creating new columns with information from load_* csv files, for the copy of current df
    for i, el in enumerate(load_columns):
        file_pattern = f'{el}'
        files = glob.glob(os.path.join(directory_path, file_pattern))

        for file_path in files:
            df = pd.read_csv(file_path)
            df_load = new_interval_load(df)
            updated_new_df[new_load_columns[i]] = df_load['Load']
    updated_new_df.fillna(0, inplace=True)
    return updated_new_df

def clean_data(df):
    # TODO: Handle missing values, outliers, etc.
    df.fillna(0, inplace=True)
    return df

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.
    id_dct = {'SP': 0, 'UK': 1, 'DE': 2, 'DK': 3, 'HU': 4, 'SE': 5, 'IT': 6, 'PO': 7, 'NL': 8}
    sp = ['SP', 'UK', 'DE', 'DK', 'HU', 'SE', 'IT', 'PO', 'NL']
    sp_green = ['green_energy_SP', 'green_energy_UK', 'green_energy_DE', 'green_energy_DK','green_energy_HU', 'green_energy_SE', 'green_energy_IT', 'green_energy_PO','green_energy_NE']
    new_load_columns = ['SP_Load', 'UK_Load', 'DE_Load', 'DK_Load', 'HU_Load', 'SE_Load', 'IT_Load', 'PO_Load', 'NE_Load']

    # the surplus of green energy
    for i, elem in enumerate(sp):
        df[elem] = df[sp_green[i]] - df[new_load_columns[i]]

    uk_all_zeros = (df['UK'].iloc[:4731] == 0).all()
    # after certain index column UK contains 0, so it is decided to not compare UK values with others after this index
    df['ID'] = df[[col for i, col in enumerate(sp) if col != 'UK' or i > 4731]].idxmax(axis=1)
    df['ID'] = df['ID'].map(id_dct)
    return df

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    output_file = df.to_csv(output_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/processed_data.csv',
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)
