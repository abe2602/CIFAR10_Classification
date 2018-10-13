import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

num_classes = 10

def showImages(class_names, img):
	plt.figure(figsize=(10,10))

	for i in range(len(class_names)):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(img[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[i])

	plt.show()

if __name__ == '__main__':
	class_names = ['airplane.jpg', 'automobile.jpg', 'bird.jpg', 'cat.jpg', 'deer.jpg', 
	               'dog.jpg', 'frog.jpg', 'horse.jpg', 'ship.jpg', 'truk.jpg']

	class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
	               'dog', 'frog', 'horse', 'ship', 'truck']

	model = keras.models.load_model('my_model2.h5')

	img = []
	for i in range (len(class_names)):
		im = cv2.imread(class_names[i])
		im = cv2.resize(im, (32, 32))
		img.append(im)

	#Mostra as imagens
	#showImages(class_names, img)

	#Normaliza as imagens
	img = np.asarray(img)
	img = img/255

	#As a predição
	predictions = model.predict(img)
	print("\n")

	#Mostra o obtido
	for i in range(len(predictions)):
		print("Quero: ", class_labels[i], "Obtido: ", class_labels[np.argmax(predictions[i])])
