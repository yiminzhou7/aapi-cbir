# aapi-cbir
Proyecto CBIR
* Archivo `CBIR_codigo_final.pynb`: el código con todas las extracciones de características probadas y los modelos probados.
* Carpeta **app-streamlit**:
  * `modelo.py`: contiene las funciones que se usarán para la búsqueda de imágenes similares. Como se ha visto en el `CBIR_codigo_final.pynb`, usaremos VGG19 + KNN.
  * `app_streamlit.py`: la interfaz gráfica para el buscador de imágenes.
* Carpeta **cifar10**: contiene el archivo con las imágenes iniciales para usarlo en el `CBIR_codigo_final.pynb`.
* Carpeta **example_images**: contiene algunas imágenes para probar con la interfaz.
* Archivo `imagenes_seleccionadas.pickle`: contiene el subconjunto de imágenes de CIFAR10 que usaremos para extraer las características y extrenar los modelos.

**Nota**: no está el archivo `características_vgg19.csv` que se usa en `app_streamlit.py` porque el archivo es demasiado grande para subirlo, pero se puede ejecutar el código `CBIR_codigo_final.pynb` en el apartado *3.1.4. VGG19 + KNN* para conseguirlo.
