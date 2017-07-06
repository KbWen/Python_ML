from keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D

np.random.seed(10)
(x_img_train, y_label_train), \
(x_img_test, y_label_test) = cifar10.load_data()

print(x_img_train.shape)
print(x_img_test[0].shape)
print(y_label_train.shape)

label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

def plot_image(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25 : num=25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx])
        title=str(i)+':'+label_dict[labels[i][0]]
        if len(prediction)>0:
            title +='+>'+label_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
# image
plot_image(x_img_train,y_label_train,[],0)
# normalize
x_img_train_nor = x_img_train.astype('float32')/255
x_img_test_nor = x_img_test.astype('float32')/255
# onehot encoding (50000,1) => (50000,10)
y_label_train_ohe = np_utils.to_categorical(y_label_train)
y_label_test_ohe = np_utils.to_categorical(y_label_test)
# print(y_label_train_ohe[:5])

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),
          activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),
          activation='relu',padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))
print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x_img_train_nor, y_label_train_ohe, validation_split=0.2,
                          epochs=10, batch_size=128,verbose=1)

# history.history('acc','val_acc')
# history.history('loss','val_loss')

scores = model.evaluate(x_img_test_nor,y_label_test_ohe, verbose=0)

prediction = model.predict_classes(x_img_test_nor)
print(scores)
print(label_dict)
print(pd.crosstab(y_label_test.reshape(-1),prediction,rownames=['label'],colnames=['predict']))
