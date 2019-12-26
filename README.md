# keras_bert_ner
基于keras和keras_bert的中文命名实体识别，搭建的网络为bert+bilstm_crf
运行main函数即可训练并使用模型
environment:
Keras version:2.2.4
keras-contrib version:2.0.8
需要下载中文预训练模型chinese_L-12_H-768_A-12放到Parameter文件夹下

## 数据集

- [Determined22/zh-NER-TF](https://github.com/Determined22/zh-NER-TF/tree/master/data_path)
- [NER_corpus_chinese](https://github.com/yaleimeng/NER_corpus_chinese)
- [ProHiryu/bert-chinese-ner](https://github.com/ProHiryu/bert-chinese-ner)