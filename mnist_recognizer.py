import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import SGD
from keras.metrics import Accuracy
from keras.utils import to_categorical


def train():
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    # draw(X,Y)
    plt.imshow(X_train[0],cmap='gray')
    plt.show()
    # 将图片二维数据转换为1维 并归一化
    X_train = X_train.reshape(60000,784) / 255.0
    X_test = X_test.reshape(10000,784)/ 255.0
    # 实际结果转换为one-hot编码
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)

    model = Sequential()
    model.add(Dense(units=256,activation='relu', input_dim=784))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.05),metrics=[Accuracy()])
    model.fit(X_train,Y_train,epochs=3000,batch_size=2048)
    # pres = model.predict(X)
    loss,accuracy = model.evaluate(X_test,Y_test)
    print(loss)
    print(accuracy)
    pass
train()
