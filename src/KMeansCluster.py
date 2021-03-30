import os
import sys

from sklearn import metrics

pwd = os.path.dirname(os.path.abspath("./"))
sys.path.append(pwd)
import h5py
import tensorflow as tf
from src.AlbertEncodeModel import AlbertEncodeModel, modeling
from src.albert.classifier_utils import get_features_test, ClassifyProcessor, \
    get_features_from_example, processor, get_features
from src.albert.hyperparameters import Hyperparamters as hp
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import jieba
from collections import Counter

tf.reset_default_graph()

def get_train_data(h5_file_path, raw_file_path, col_name):
    """
    因为直接从excel中读取数据并转化为bert可接受的数据格式的过程比较缓慢，所以将转换后的数据保存在h5py中
    :param h5_file_path:
    :param raw_file_path:
    :param col_name:
    :return:
    """
    print("h5file path:", h5_file_path)

    h5_data_file = h5_file_path
    input_ids_key = "input_ids"
    input_masks_key = "input_masks"
    segment_ids_key = "segment_ids"
    h5file = h5py.File(h5_data_file, "a")
    h5Keys = set(h5file.keys())
    if input_ids_key in h5Keys:
        input_ids = h5file[input_ids_key].value
        input_masks = h5file[input_masks_key].value
        segment_ids = h5file[segment_ids_key].value
    else:
        input_ids, input_masks, segment_ids, _ = get_features(raw_file_path, col_name)
        h5file.create_dataset(input_ids_key, data=input_ids)
        h5file.create_dataset(input_masks_key, data=input_masks)
        h5file.create_dataset(segment_ids_key, data=segment_ids)
        print("从h5：{}文件中获取数据成功".format(h5_data_file))
    h5file.close()

    return input_ids, input_masks, segment_ids


def select(data, ids):
    return [data[i] for i in ids]

class AlbertKMeansCluster(object):

    data_pre = os.path.join(pwd, "data")
    # 聚类数量
    CLUSTER_NUM = 2
    # 每类保留的单词数量
    RESERVE_NUM = 20
    def __init__(self, small = False):
        self.small = small
        self.encoder, self.sess = self.load_model()

        if not self.small:
            self.data_file = os.path.join(pwd, 'data/data.xlsx')
            self.ouput_data = os.path.join(pwd, 'data/data_out.xlsx')
        else:
            self.data_file = os.path.join(pwd, 'data/small_data.xlsx')
            self.ouput_data = os.path.join(pwd, 'data/small_data_out.xlsx')

    def load_model(self):
        sess = tf.Session()
        encoder = AlbertEncodeModel(is_training=False)
        sess.run(tf.global_variables_initializer())
        return encoder, sess

    def get_sentence_emb(self, input_ids, input_masks, input_segments):
        data_len = len(input_ids)
        num_batchs = int(data_len / hp.batch_size) + 1
        indexs = np.arange(data_len)
        embeds_list = []
        print(data_len)
        for i in range(num_batchs):
            print("编码中：{} / {}".format(i, num_batchs - 1))
            start_id = i * hp.batch_size
            end_id = min((i + 1) * hp.batch_size, data_len)
            i1 = indexs[start_id: end_id]
            input_id_ = select(input_ids, i1)
            input_mask_ = select(input_masks, i1)
            input_segment_ = select(input_segments, i1)
            fd = {
                self.encoder.input_ids: input_id_,
                self.encoder.input_masks: input_mask_,
                self.encoder.input_segment_ids: input_segment_
            }
            embeds = self.sess.run([self.encoder.ouput_layer], feed_dict=fd)[0]
            embeds_list.extend(embeds)
        return embeds_list

    def get_embedding_data(self):
        """
        因为直接从excel中读取数据并转化为bert可接受的数据格式的过程比较缓慢，所以将转换后的数据保存在h5py中
        :return:
        """
        if not self.small:
            h5_file_path = os.path.join(pwd, 'data/embed_data.hdf5')
        else:
            h5_file_path = os.path.join(pwd, 'data/small_embed_data.hdf5')

        print("h5file path:", h5_file_path)
        h5_data_file = h5_file_path
        embeddings_key = "embeddings"
        h5file = h5py.File(h5_data_file, "a")
        h5Keys = set(h5file.keys())
        if embeddings_key in h5Keys:
            embeddings_val = h5file[embeddings_key].value
        else:
            input_ids, input_masks, segment_ids = self.load_data()
            embeddings_val = self.get_sentence_emb(input_ids, input_masks, segment_ids)
            h5file.create_dataset(embeddings_key, data=embeddings_val)
            print("从h5：{}文件中获取词向量数据成功".format(h5_data_file))
        h5file.close()
        return embeddings_val

    def do_cluster(self):
        """
        进行聚类
        :return:
        """
        embeds_list = self.get_embedding_data()
        kmenas = KMeans(n_clusters=self.CLUSTER_NUM, random_state=9).fit(embeds_list)
        res = kmenas.labels_
        df = pd.read_excel(self.data_file)
        # 计算轮廓系数，用以评价聚类指标，越大越好
        # https://blog.csdn.net/qq_27825451/article/details/94436488
        score = metrics.silhouette_score(embeds_list, res)
        print("聚类完成，评分：", score)
        df['label'] = res
        df.to_excel(self.ouput_data)
        word_cnt_df = self.get_word_count(df)
        word_cnt_df.to_excel(os.path.join(self.data_pre, "word_cnt_res.xlsx"))

    def get_word_count(self, df):
        label_col = "label"
        label_group = df.groupby(label_col)
        labels = set(df[label_col].to_list())
        word_cnt_df = pd.DataFrame()
        waste_words = self.get_stop_words()
        for label in labels:
            label_df = label_group.get_group(label)
            all_text = label_df['content'].sum()
            top_words_df = self.do_word_count(all_text, waste_words)
            top_words_df['label'] = label
            if word_cnt_df.empty:
                word_cnt_df = top_words_df
            else:
                word_cnt_df = pd.concat([word_cnt_df, top_words_df], axis=1)

        return word_cnt_df

    def get_stop_words(self):
        with open(os.path.join(self.data_pre, "waste_words.txt"),'r', encoding="utf-8") as r:
            waste_words = r.readlines()
        waste_words = list(map(lambda x: x.strip(), waste_words))
        return waste_words

    def do_word_count(self, text, waste_words):
        words = jieba.cut(text)
        words = list(filter(lambda x: len(x) >= 2, words))
        words = list(filter(lambda x: x not in waste_words, words))
        words_df = pd.DataFrame(words)
        words_df.columns = ['words']
        words_df['cnt'] = 1
        words_df = words_df.groupby('words')['cnt'].sum().reset_index()
        words_df.sort_values(by='cnt', ascending=False, inplace=True)
        top_words_df = words_df.head(self.RESERVE_NUM).reset_index()
        top_words_df = top_words_df.loc[:, ['words', 'cnt']]
        return top_words_df

    def load_data(self):
        if not self.small:
            h5_file_path = os.path.join(pwd, 'data/lx_data.hdf5')
        else:
            h5_file_path = os.path.join(pwd, 'data/small_lx_data.hdf5')
        input_ids, input_masks, segment_ids = get_train_data(h5_file_path, self.data_file, "content")
        return input_ids, input_masks, segment_ids

if __name__ =='__main__':
    ak = AlbertKMeansCluster(small=True)
    ak.do_cluster()