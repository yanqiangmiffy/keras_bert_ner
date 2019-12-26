#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: quincy qiang 
@license: Apache Licence 
@file: predict.py 
@time: 2019/12/25
@software: PyCharm 
"""
import pandas as pd
from model.bertbilstmcrf import bert_bilstm_crf
from utils import get_entity, get_ORG_entity
from tqdm import tqdm

max_seq_length = 80
batch_size = 16
epochs = 25
lstmDim = 64
model = bert_bilstm_crf(max_seq_length, batch_size, epochs, lstmDim)


def get_org(sentence):
    demo_sent = list(sentence.strip())
    tag = model.ModelPredict(sentence)
    orgs = get_ORG_entity(tag, demo_sent)
    return orgs


# df_coorp = pd.read_excel('合作.xlsx')
# df_coorp['新闻标题'] = df_coorp['新闻标题'].astype(str)
# res = []
# for title in tqdm(df_coorp['新闻标题']):
#     res.append(";".join(get_org(title[:79])))
# df_coorp['title_orgs'] = res
#
# res = []
# df_coorp['新闻内容（部分）'] = df_coorp['新闻内容（部分）'].astype(str)
#
# for title in tqdm(df_coorp['新闻内容（部分）']):
#     res.append(";".join(get_org(title[:79])))
# df_coorp['abstract_orgs'] = res
#
# df_coorp.to_excel('合作_v2.xlsx', index=None)
# # df_coorp['abstract_orgs'] = df_coorp['新闻内容（部分）'].apply(lambda x: ' '.join(get_org(x)))


df_coorp = pd.read_excel('供应商.xlsx')
df_coorp['新闻标题'] = df_coorp['新闻标题'].astype(str)
res = []
for title in tqdm(df_coorp['新闻标题']):
    res.append(";".join(get_org(title[:79])))
df_coorp['title_orgs'] = res

res = []
df_coorp['新闻内容（部分）'] = df_coorp['新闻内容（部分）'].astype(str)

for title in tqdm(df_coorp['新闻内容（部分）']):
    res.append(";".join(get_org(title[:79])))
df_coorp['abstract_orgs'] = res

df_coorp.to_excel('供应商_v2.xlsx', index=None)
# df_coorp['abstract_orgs'] = df_coorp['新闻内容（部分）'].apply(lambda x: ' '.join(get_org(x)))