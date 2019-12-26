from model.bertbilstmcrf import bert_bilstm_crf
from model import dealwithdata
import ipykernel
from utils import get_entity
from sklearn.utils import shuffle
if __name__ == "__main__":
    # 数据
    train_path = "data/train.txt"
    test_path = "data/test.txt"
    train_data = dealwithdata.PreProcessData(train_path)
    test_data = dealwithdata.PreProcessData(test_path)
    print(len(train_data[0]))
    add_data = dealwithdata.PreProcessData('data/other/corporation.txt')
    train_data[0] = train_data[0] + add_data[0]
    train_data[1] = train_data[1] + add_data[1]
    print(len(train_data[0]))
    #
    # 模型
    max_seq_length = 100
    batch_size = 16
    epochs = 25
    lstmDim = 64
    model = bert_bilstm_crf(max_seq_length, batch_size, epochs, lstmDim)
    model.TrainModel(train_data, test_data, 'tmp/keras_bert.hdf5')

    # 测试
    while 1:
        sentence = input('please input sentence:\n')
        demo_sent = list(sentence.strip())
        tag = model.ModelPredict(sentence, 'tmp/keras_bert.hdf5')
        print(tag)
        PER, LOC, ORG = get_entity(tag, demo_sent)
        print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
