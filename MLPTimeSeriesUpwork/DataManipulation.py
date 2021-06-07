import pandas as pd
import numpy as np


class DataManipulation:

    def read_dataset(self, dataset_file_path):
        all_dfs = pd.read_excel(dataset_file_path, sheet_name=None)
        all_dfs.keys()
        sheet_key_list = list( all_dfs.keys())
        columns_name = ["Product Name", "Category","Unit price", "Quantity", "Date", "sales", "Holiday", "Temprature"]

        dataset_df = all_dfs[sheet_key_list[0]]

        for index in range(1, len(sheet_key_list)):
            tmp_df = all_dfs[sheet_key_list[index]]
            tmp_df.columns = columns_name
            dataset_df = dataset_df.append(tmp_df, ignore_index=True)

        table = pd.pivot_table(dataset_df, values="sales", index=["Date", "Holiday", "Temprature"], columns=["Category"], aggfunc=np.sum, fill_value=0)
        table.reset_index(inplace=True)

        columns_name = ["Date", "Holiday", "Temprature"]
        mean_df = table[columns_name].groupby(["Date"]).mean()

        columns_name = ["Date", "BEAUTY", "EVERY DAY ESSENTIALS", "HOME HEALTH CARE", "MOM AND BABY", "Medication", "Vitamins and Nutrition"]
        sum_df = table[columns_name].groupby(["Date"]).sum()

        df = sum_df.merge(mean_df, left_on='Date', right_on='Date', how="inner")

        columns_name.remove("Date")

        df.reset_index(inplace=True)

        return df, columns_name

    def split_sequences(self, sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):

            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out

            if out_end_ix > len(sequences):
                break

            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)