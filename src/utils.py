# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:43:39 2018

@author: cm
"""

import time
import numpy as np
import pandas as pd
import os
import sys
pwd = os.path.dirname(os.path.abspath('./'))
sys.path.append(pwd)
up_pwd = os.path.dirname(os.path.abspath('../'))
sys.path.append(up_pwd)

def cut_list(data, size):
    """
    data: a list
    size: the size of cut
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data) - 1) // size + 1)]


def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def select(data, ids):
    return [data[i] for i in ids]


def load_txt(file):
    with open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        lines = list(filter(lambda x: len(x) >= 1, lines))
        print("Load data from file (%s) finished !" % file)
    return lines


def save_txt(file, lines):
    lines = [l + '\n' for l in lines]
    with  open(file, 'w+', encoding='utf-8') as fp:  # a+添加
        fp.writelines(lines)
    return "Write data to txt finished !"

def load_useful_ft_set():
    with open(os.path.join(up_pwd, "data/m_law_label.txt"), 'r', encoding='utf-8') as r:
        ft_lists = r.readlines()
    ft_set = set(map(lambda x: int(x.strip()), ft_lists))
    return ft_set

def load_csv(file, header=None):
    data = pd.read_excel(file)  # ,encoding='gbk'
    data.fillna("|", inplace=True)
    return data


def save_csv(dataframe, file, header=True, index=None, encoding="gbk"):
    return dataframe.to_csv(file,
                            mode='w+',
                            header=header,
                            index=index,
                            encoding=encoding)


def save_excel(dataframe, file, header=True, sheetname='Sheet1'):
    return dataframe.to_excel(file,
                              header=header,
                              sheet_name=sheetname)


def load_excel(file, header=0):
    return pd.read_excel(file,
                         header=header,
                         )


DEFAULT_VOC_LABEL_FILE = "./data/vocabulary_label.txt"


def load_vocabulary(file_vocabulary_label):
    """
    Load vocabulary to dict
    """
    vocabulary = load_txt(file_vocabulary_label)
    dict_id2label, dict_label2id = {}, {}
    for i, l in enumerate(vocabulary):
        dict_id2label[str(i)] = str(l)
        dict_label2id[str(l)] = str(i)
    return dict_id2label, dict_label2id


def shuffle_two(a1, a2):
    """
    随机打乱a1和a2两个
    """
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    a1_ = [a1[l] for l in ran]
    a2_ = [a2[l] for l in ran]
    return a1_, a2_


def map_train_id():
    IMPORT_FILE_DIR = "./important_data/"
    fdqj = pd.read_csv(os.path.join(IMPORT_FILE_DIR, "fdlxqj.csv"))
    zdqj = pd.read_csv(os.path.join(IMPORT_FILE_DIR, "zdlxqj.csv"))
    id_map = {}
    id_map = get_train_id(fdqj, id_map)
    id_map = get_train_id(zdqj, id_map)
    return id_map


def get_train_id(df, id_map):
    for item in df.itertuples():
        id_map[item.sid] = item.id
    return id_map



def combine_data():
    data_file = os.path.join(pwd, 'data/')
    afh = pd.read_excel(os.path.join(data_file, "案发后-1.xlsx"))
    other = pd.read_excel(os.path.join(data_file, "其它-1.xlsx"))
    afs = pd.read_excel(os.path.join(data_file, "案发时-1.xlsx"))
    pc = pd.read_excel(os.path.join(data_file, "赔偿-1.xlsx"))
    jd = pd.read_excel(os.path.join(data_file, "鉴定-1.xlsx"))
    dict_id2label, dict_label2id = load_vocabulary(os.path.join(data_file, "ef_label.txt"))
    data_df = pd.concat([other, afh, afs, pc, jd], axis=0)
    data_df.columns = ['label', "ef"]
    data_df = data_df.loc[data_df['label'] != 'label', :]
    data_df = data_df.loc[data_df['label'] > 0, :]
    data_df['label'] = data_df['label'].apply(lambda x: dict_id2label[str(x - 1)])
    show_base_info(data_df)
    print(data_df.shape)
    sample_save_data(data_df, "ef", data_file)


def sample_save_data(data, file_tail = 'default', FILE_PATH = "./"):
    """
    划分数据集并保存数据
    :param data:
    :return:
    """

    data = data.reset_index()
    train_data = data.sample(frac=0.8)
    left_data = data.drop(train_data.index, axis=0)
    # val_data = left_data.sample(frac=0.5)
    # test_data = left_data.drop(val_data.index, axis=0)

    show_base_info(train_data)
    show_base_info(left_data)

    # train_data.to_csv(os.path.join(FILE_PATH, "train-{}.csv".format(file_tail)), index=False)
    # left_data.to_csv(os.path.join(FILE_PATH, "test-{}.csv".format(file_tail)), index=False)

def show_base_info(data_df):
    data_df['cnt'] = 1
    dg = data_df.groupby(by='label')['cnt'].sum().reset_index()
    print(dg.head(10))

if __name__ == '__main__':
    # id_map = map_train_id()
    combine_data()
