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
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten

def train():
    (X_train,Y_train),(X_test,Y_test) = mnist.load_data()
    # draw(X,Y)
    plt.imshow(X_train[0],cmap='gray')
    plt.show()
    # 将图片二维数据转换为1维 并归一化
    X_train = X_train.reshape(60000,28,28,1) / 255.0
    X_test = X_test.reshape(10000,28,28,1)/ 255.0
    # 实际结果转换为one-hot编码
    Y_train = to_categorical(Y_train,10)
    Y_test = to_categorical(Y_test,10)

    model = Sequential()
    # 卷积层
    model.add(Conv2D(filters=6,kernel_size=(5,5), strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu'))
    # 池化层
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16,kernel_size=(5,5), strides=(1,1),padding='valid',activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    # 展开池化后的数据
    model.add(Flatten())
    # 全连接层
    model.add(Dense(units=120,activation='relu'))
    model.add(Dense(units=84,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    # 使用分类交叉熵作为损失函数 
    model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.05),metrics=[Accuracy()])
    model.fit(X_train,Y_train,epochs=5000,batch_size=2048)
    # pres = model.predict(X)
    loss,accuracy = model.evaluate(X_test,Y_test)
    print(loss)
    print(accuracy)
    pass
train()
