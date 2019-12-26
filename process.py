#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: quincy qiang 
@license: Apache Licence 
@file: process.py 
@time: 2019/12/25
@software: PyCharm 
"""
import os

aug_file = open('data/other/aug_data.txt', 'w', encoding='utf-8')

with open('data/other/example.train', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        aug_file.write(line)


with open('data/other/example.test', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        aug_file.write(line)

with open('data/other/example.dev', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        aug_file.write(line)


with open('data/other/train_data', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        aug_file.write(line)

with open('data/other/test_data', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        aug_file.write(line)

aug_file.close()
