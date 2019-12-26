#! -*- coding:utf-8 -*-

def PreProcessData(path):
    sentences = []
    tags = []
    with open(path, encoding="utf-8") as data_file:
        for sentence in data_file.read().strip().split('\n\n'):
            _sentence = ""
            tag = []
            for word in sentence.strip().split('\n'):
                content = word.strip().split()
                _sentence += content[0]
                tag.append(content[1])
            sentences.append(_sentence)
            tags.append(tag)
    data = [sentences, tags]
    return data
