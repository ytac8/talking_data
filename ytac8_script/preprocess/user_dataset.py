import pandas as pd
import pickle

if __name__ == "__main__":
    with open('../data/pickle/train.csv.pkl', mode='rb') as f:
        train_df = pickle.load(f)

    user_df = pd.read_csv('../data/preprocessed/user_list.csv', header=None)
    user_list = user_df.iloc[:, 0]
    user_list = user_list.tolist()

    for user in user_list:
        a = train_df[train_df['ip'] == user]
        file_name = "../data/preprocessed/user_data/data_" + str(user) + ".csv"
        a.to_csv(file_name, index=None)
