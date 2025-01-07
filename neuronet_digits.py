import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import tensorflow_datasets as tfds
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Загружаем датасет с цифрами
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Создаём модель
model = tf.keras.Sequential([
    Flatten(input_shape=(28,28)),   # Входной слой для изображений 28x28
    Dense(128, activation='relu'),   # Скрытый слой со 128 нейронами
    Dense(10, activation='softmax')  # Выходной слой с 10 нейронами (по 1 нейрону для каждой цифры)
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
print(model.evaluate(X_test, y_test))
print(model.summary())

n = 1
x = np.expand_dims(X_test[n], axis=0)
res = model.predict(x)
print(res)
print('Распознанная цифра:', np.argmax(res))
plt.imshow(X_test[n], cmap=plt.cm.binary)
plt.show()