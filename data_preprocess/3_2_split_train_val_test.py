import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser(description='split train and validation set')
parser.add_argument('-b', '--base_dataset_path', type=str, default=None, help="dataset root path", required=True)
parser.add_argument('-s', '--speaker', type=str, default='Default Speaker Name', required=True)

args = parser.parse_args()

DATASET_PATH = os.path.join(args.base_dataset_path, args.speaker)
TMPCSV_PATH = os.path.join(DATASET_PATH, "tmp", "intermediate_csv")


if __name__ == "__main__":
    train_test_ratio = 0.8
    dr_all_csv = TMPCSV_PATH

    print(f"Fn ``split_train_val'', train_val_ratio={train_test_ratio}")
    idle_num = 13
    # Since when generating data samples, stride=5, then after at least 13 data samples,
    # the train and validation set would completely share no frames

    ls_all_csv = sorted(os.listdir(dr_all_csv))
    ls_all_csv = [os.path.join(dr_all_csv, i) for i in ls_all_csv if i.startswith("tmp")]
    print(dr_all_csv)
    ls_train_df, ls_test_df, ls_idle_df = [], [], []
    for csv_path in ls_all_csv:
        df = pd.read_csv(csv_path)
        total_num = len(df)
        train_test_split = int(total_num*train_test_ratio)
        ls_train_df.append(df.iloc[:train_test_split])

        idle_df = df.iloc[train_test_split: train_test_split + idle_num]
        # change the ``dataset'' column in idle_df to be ``idle''
        idle_df.loc[:, "dataset"] = "idle"
        ls_idle_df.append(idle_df)

        test_df = df.iloc[train_test_split + idle_num:]
        # change the ``dataset'' column in test_df to be ``dev''
        test_df.loc[:, "dataset"] = "val"
        ls_test_df.append(test_df)

        print(f"file: {os.path.basename(csv_path)}, total_num: {total_num}, "
              f"train: {train_test_split}, test: {total_num - train_test_split}")
    ans_df = pd.concat([pd.concat(ls_train_df), pd.concat(ls_idle_df), pd.concat(ls_test_df)])
    ans_df.to_csv(os.path.join(DATASET_PATH, f"clips.csv"), index=False)


