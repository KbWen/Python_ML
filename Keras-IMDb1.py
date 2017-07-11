import io
import sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
# encoding UTF-8
import urllib.request
import os
import tarfile
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
import re
# import imdb data
url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
filepath = "dataimdb/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url,filepath)
    print('download:',result)

if not os.path.exists('dataimdb/aclImdb'):
    tfile = tarfile.open('dataimdb/aclImdb_v1.tar.gz', 'r:gz')
    result = tfile.extractall('dataimdb/')
# remove html tag
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)
# This is one way to store data in pos files and neg files
def read_files(filetype):
    path = 'dataimdb/aclImdb/'
    file_list = []
    pos_path = path + filetype + '/pos/'
    for i in os.listdir(pos_path):
        file_list+=[pos_path+i]
    neg_path = path + filetype + '/neg/'
    for i in os.listdir(neg_path):
        file_list+=[neg_path+i]
    print('read',filetype, 'files:',len(file_list))
    all_labels = ([1]*12500+[0]*12500)
    all_texts = []
    for i in file_list:
        with open(i,encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels,all_texts
# put in train data
y_train,train_text = read_files('train')
y_test,test_text = read_files('test')
# word to number and word embeddings
token = Tokenizer(num_words=3000)
token.fit_on_texts(train_text)
print(token.document_count)
print(token.word_index)

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
x_train = sequence.pad_sequences(x_train_seq, maxlen=300)
x_test = sequence.pad_sequences(x_test_seq, maxlen=300)

print(train_text[0])
print(x_train_seq[0])
print(x_train[0])

# Deep learning LSTM
model = Sequential()
model.add(Embedding(output_dim=64, input_dim=3000, input_length=300))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()
# optimizer
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_train, y_train, batch_size=100,epochs=10,verbose=2,
                          validation_split=0.2)
# scores
scores = model.evaluate(x_test,y_test,verbose=1)
print(scores[1])
# test data from IMDb spider-man
input_text= "Yes. This may be a great coming of age movie, and a great superhero movie but not the greatest spider-man movie. I'm not being stubborn here about the classic. I've seen reviews saying it is the best Spider- man movie ever. I can't agree with that because the villain and some supporting characters are so underdeveloped. Tom Holland tries a ultimate spider-man style but in the end all we get is the Kung-fu Panda version of spider-man. It was fun in civil war because it was short but here he keeps whining about it almost every minute when he was Peter Parker. In overall, this is just another MCU popcorn movie. It's entertaining as hell, good to re-watch on Blu-ray and keep it in your MCU collection. BTW, Tony stark was awesome once again. "

input_seq = token.texts_to_sequences([input_text])
pad_input_seq = sequence.pad_sequences(input_seq,maxlen=300)
predict_result = model.predict_classes(pad_input_seq)
print(predict_result)
