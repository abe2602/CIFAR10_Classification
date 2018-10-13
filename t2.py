import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
epochs = 300
batch_size = 128
shuffle = True

if __name__ == '__main__':
  
    #Load nos dados
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    test_labels = keras.utils.to_categorical(test_labels, num_classes)
    train_labels = keras.utils.to_categorical(train_labels, num_classes)

    #Cria o modelo
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', padding='same', input_shape=(32, 32, 3)))
    
    model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
   
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
   
    model.add(keras.layers.Dense(10, activation='softmax'))

    #Compila o modelo
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    #Treina o modelo
    model.fit(train_images, train_labels,
              batch_size=batch_size,
              shuffle=shuffle,
              epochs=epochs,
              validation_data=(test_images, test_labels))

    #Avalia o modelo
    loss, acc = model.evaluate(test_images, test_labels)

    #Salva o modelo
    model.save("my_model1.h5")

print('Accuracy: %.3f' % acc)
print("Modelo criado com sucesso")