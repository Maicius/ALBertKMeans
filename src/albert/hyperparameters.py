# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:23:12 2018

@author: cm
"""

import os
import sys

pwd = os.path.dirname(os.path.abspath("./"))
sys.path.append(pwd)

class Hyperparamters:
    # Train parameters
    num_train_epochs = 5
    print_step = 100
    batch_size = 32
    summary_step = 100
    num_saved_per_epoch = 3
    max_to_keep = 20

    epsilon = 1e-9

    # Train/Test data
    data_dir = os.path.join(pwd, 'data')
    train_data = 'train-zzm.csv'
    val_data = 'val-zzm.csv'
    test_data = 'test-zzm.csv'

    train_data_raw = 'train-zzm.csv'
    val_data_raw = 'val-zzm.csv'
    test_data_raw = 'test-zzm.csv'

    file_model = 'model/model_01'
    # Optimization parameters
    warmup_proportion = 0.1
    use_tpu = None
    do_lower_case = True
    learning_rate = 5e-5

    # TextCNN parameters
    num_filters = 128
    filter_sizes = [2, 3, 4]
    embedding_size = 384 # 12 * 32 and 64 * 6
    keep_prob = 0.5

    seq_len = 128

    # ALBERT
    model = 'albert_small_zh_google'
    bert_path = os.path.join(pwd, model)
    vocab_file = os.path.join(pwd, model, 'vocab_chinese.txt')
    init_checkpoint = os.path.join(pwd, model, 'albert_model.ckpt')
    saved_model_path = os.path.join(pwd, 'model')

