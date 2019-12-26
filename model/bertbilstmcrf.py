import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from keras_bert import Tokenizer
from keras_bert import load_trained_model_from_checkpoint
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

eraly_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='auto')
# Reducing the Learning Rate if result is not improving.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',
                              verbose=1)


class bert_bilstm_crf:
    def __init__(self, max_seq_length, batch_size, epochs, lstm_dim):
        self.label = {}
        self._label = {}
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lstmDim = lstm_dim
        self.LoadLabel()
        self.model = self.Model()

    # 抽取的标签
    def LoadLabel(self):
        # label
        label_path = ".//parameter//tag_dict.txt"
        f_label = open(label_path, 'r+', encoding='utf-8')
        for line in f_label:
            content = line.strip().split()
            self.label[content[0].strip()] = content[1].strip()
            self._label[content[1].strip()] = content[0].strip()
        # dict
        self.vocab = {}
        vocab_path = ".//parameter//chinese_L-12_H-768_A-12//vocab.txt"
        with open(vocab_path, 'r+', encoding='utf-8') as f_vocab:
            for line in f_vocab.readlines():
                self.vocab[line.strip()] = len(self.vocab)

    # 模型
    def Model(self):
        model_path = ".\\parameter\\chinese_L-12_H-768_A-12\\"
        bert = load_trained_model_from_checkpoint(
            model_path + "bert_config.json",
            model_path + "bert_model.ckpt",
            seq_len=self.max_seq_length
        )
        # make bert layer trainable
        for layer in bert.layers:
            layer.trainable = True
        x1 = Input(shape=(None,))
        x2 = Input(shape=(None,))
        bert_out = bert([x1, x2])
        lstm_out = Bidirectional(LSTM(self.lstmDim,
                                      return_sequences=True,
                                      dropout=0.2,
                                      recurrent_dropout=0.2))(bert_out)
        crf_out = CRF(len(self.label), sparse_target=True)(lstm_out)
        model = Model([x1, x2], crf_out)
        model.summary()
        model.compile(
            optimizer=Adam(1e-4),
            loss=crf_loss,
            metrics=[crf_accuracy]
        )
        return model

    # 预处理输入数据
    def PreProcessInputData(self, text):
        tokenizer = Tokenizer(self.vocab)
        word_labels = []
        seq_types = []
        for sequence in text:
            code = tokenizer.encode(first=sequence, max_len=self.max_seq_length)
            word_labels.append(code[0])
            seq_types.append(code[1])
        return word_labels, seq_types

    # 预处理结果数据
    def PreProcessOutputData(self, text):
        tags = []
        for line in text:
            tag = [0]
            for item in line:
                tag.append(int(self.label[item.strip()]))
            tag.append(0)
            tags.append(tag)

        pad_tags = pad_sequences(tags, maxlen=self.max_seq_length, padding="post", truncating="post")
        result_tags = np.expand_dims(pad_tags, 2)
        return result_tags

    # 训练模型
    def TrainModel(self, train_data, test_data,model_path):
        input_train, result_train = train_data
        input_test, result_test = test_data
        # 训练集
        input_train_labels, input_train_types = self.PreProcessInputData(input_train)
        result_train = self.PreProcessOutputData(result_train)
        # 测试集
        input_test_labels, input_test_types = self.PreProcessInputData(input_test)
        result_test = self.PreProcessOutputData(result_test)
        history = self.model.fit(x=[input_train_labels, input_train_types],
                                 y=result_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=[[input_test_labels, input_test_types], result_test],
                                 verbose=1,
                                 callbacks=[eraly_stop, reduce_lr],
                                 shuffle=True)
        self.model.save(model_path)
        self.model_improve_process(history)
        return

    # 预测结果
    def Id2Label(self, ids):
        result = []
        for id in ids:
            result.append(self._label[str(id)])
        return result

    def Vector2Id(self, tags):
        result = []
        for tag in tags:
            result.append(np.argmax(tag))
        return result

    def ModelPredict(self, sentence,model_apth):
        labels, types = self.PreProcessInputData([sentence])
        self.model.load_weights(model_apth)
        tags = self.model.predict([labels, types])[0]
        result = []
        for i in range(1, len(sentence) + 1):
            result.append(tags[i])
        result = self.Vector2Id(result)
        tag = self.Id2Label(result)
        return tag

    # 演示训练过程，用于观察拟合程度
    def model_improve_process(self, history):
        history_dict = history.history
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'yo', label='Training loss')
        plt.plot(epochs, val_loss, 'y', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend
        plt.show()
