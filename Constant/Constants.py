# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : Constant.py
# @Project: GOODKT
# @Comment :
Dpath = '../../Dataset'
datasets = {
    'assist2009' : 'assist2009',
    'assist2012' : 'assist2012',
    'assist2017' : 'assist2017',
    'assistednet': 'assistednet',
}

# question number of each dataset
students = {
    'assist2009' : 8139,
    'assist2012' : 1,
    'assist2017' : 1,
    'assistednet': 7572,
}

numbers = {
    'assist2009' : 16891,
    'assist2012' : 37125,
    'assist2017' : 3162,
    'assistednet': 10795,
}

skill = {
    'assist2009' : 101,
    'assist2012' : 188,
    'assist2017' : 102,
    'assistednet': 1676,   # 188
}

DATASET = datasets['assistednet']
NUM_OF_QUESTIONS = numbers['assistednet']
H = 'ednet'

MAX_STEP = 50
BATCH_SIZE = 128
LR = 0.001
EPOCH = 20
EMB = 256
HIDDEN = 128  # sequence model's
kd_loss = 5.00E-06
LAYERS = 1

RUN_ID = "Proposal_Constrastive_Graph_Attention"

user_num = students['assistednet']
item_num = numbers['assistednet']*2
