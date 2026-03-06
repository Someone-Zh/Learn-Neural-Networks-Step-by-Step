import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.losses import mean_squared_error
from keras.optimizers import SGD
from keras.metrics import Accuracy
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten

from keras.preprocessing import sequence

x_train, y_train, x_test, y_test = shopping_data.load_data()

print('x_train.shape:', x_train.shape)
print('y_train.shape:', y_train.shape)
print('x_test.shape:', x_test.shape)
print('y_test.shape:', y_test.shape)
print(x_train[0])
print(y_train[0])

vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)
print(word_index)
print('词典总词数：', vocalen)

x_train_index = shopping_data.word2Index(x_train, word_index)
x_test_index = shopping_data.word2Index(x_test, word_index)

maxlen = 25
x_train_index = sequence.pad_sequences(x_train_index, maxlen=maxlen)
x_test_index = sequence.pad_sequences(x_test_index, maxlen=maxlen)
model = Sequential()
model.add(Embedding(trainable=False, input_dim=vocalen, output_dim=300, input_length=maxlen))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_index, y_train,
          batch_size=512,
          epochs=200)
score, acc = model.evaluate(x_test_index, y_test)

print('Test score:', score)
print('Test accuracy:', acc)

pass
