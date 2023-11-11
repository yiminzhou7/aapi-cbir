import pickle
import numpy as np
import pandas as pd
import cv2
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg16 import preprocess_input

from sklearn.neighbors import NearestNeighbors


# Cargar el diccionario desde el archivo pickle
with open('../imagenes_seleccionadas.pickle', 'rb') as f:
    diccionario_cargado = pickle.load(f)

# Acceder a las imágenes y etiquetas
imagenes_seleccionadas = diccionario_cargado['imagenes']
etiquetas_seleccionadas = diccionario_cargado['etiquetas']
print('Ok cargado')


# Mostrar las imágenes con menos ruido
def filtro_laplaciano(imagen):
    imagen_laplaciana = cv2.Laplacian(imagen, cv2.CV_64F)
    imagen_nitida = imagen - 0.5 * imagen_laplaciana  # Ajusta el factor según sea necesario
    imagen_nitida = np.clip(imagen_nitida, 0, 255).astype(np.uint8)
    return imagen_nitida

def vgg19():
    base_model = VGG19(weights='imagenet', include_top=False)
    return base_model


# Función para cargar y preprocesar imágenes según VGG19
def cargar_y_preprocesar_imagen(imagen):
    imagen_1 = cv2.resize(imagen, (224, 224))  
    imagen_2 = np.expand_dims(imagen_1, axis=0) 
    imagen_3 = preprocess_input(imagen_2)  
    return imagen_3

# Crear el modelo KNN y ajustarlo a las características
def buscar_imagenes_KNN(todas_caracteristicas, caracteristicas_query, k_vecinos, metrica, n_imagenes_similares):
    knn_model = NearestNeighbors(n_neighbors=k_vecinos, metric=metrica)
    knn_model.fit(todas_caracteristicas)
    _, indices = knn_model.kneighbors(caracteristicas_query.reshape(1, -1), n_neighbors=n_imagenes_similares)

    imagenes_similares = [] 
    clase_similares = []  

    for i in indices[0]:
        imagen_similar = imagenes_seleccionadas[i]
        imagenes_similares.append(imagen_similar)
        clase = etiquetas_seleccionadas[i]
        clase_similares.append(clase)

    return imagenes_similares, clase_similares

