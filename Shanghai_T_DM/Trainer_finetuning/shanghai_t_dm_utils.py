import pandas as pd
import glob
import os
import numpy as np

def read_dataset(type="T1"):
    TEST_INPUT_FOLDER = f"../../datasets/Shanghai_{type}DM"

    dataframes = []
    file_names = []

    excel_files = glob.glob(os.path.join(TEST_INPUT_FOLDER, "*.xls")) + glob.glob(os.path.join(TEST_INPUT_FOLDER, "*.xlsx"))

    for file in excel_files:
        df = pd.read_excel(file)
        if 'CGM (mg / dl)' in df.columns:
            df = df[['Date', 'CGM (mg / dl)']]
        else:
            df = df[['Date', 'CGM ']]

        if df.isna().any().any():
            print(f"DataFrame {file} has columns with NaN values:")
            print(df.isna().sum())
            continue

        dataframes.append(df.sort_values(by='Date').rename(columns={
            'CGM (mg / dl)': 'CGM',
            'CGM ': 'CGM'
        }))

        file_names.append(file.split('.')[-2].split('/')[-1])


    return dataframes, file_names
