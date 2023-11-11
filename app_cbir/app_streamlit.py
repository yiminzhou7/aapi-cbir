import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd


from modelo import vgg19, buscar_imagenes_KNN, cargar_y_preprocesar_imagen, filtro_laplaciano


def main():
    # Título centrado
    st.markdown("<h1 style='text-align: center;'>BUSCADOR DE IMÁGENES</h1>", unsafe_allow_html=True)


    # hacemos que estas variables persistan durante la sesión, es decir, solo
    # se carga una vez (Lo hacemos así porque streamlit ejecuta el código cada
    # vez que el usuario realiza algo en la web, por lo que estas variables se cargarán 
    # cada vez que el usuario haga algo).
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv('caracteristicas_vgg19.csv')
    if 'caracteristicas_vgg19' not in st.session_state:
        st.session_state.caracteristicas_vgg19 = st.session_state.df.values
    if 'modelo_vgg19' not in st.session_state:
        st.session_state.modelo_vgg19 = vgg19()

    img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])

    # El usuario carga una imagen
    if img_file_buffer is not None:
        image_query = np.array(Image.open(img_file_buffer))
        texto = "Imagen de consulta"
        st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: larger;'>{texto}</div>", unsafe_allow_html=True) 

        col1, col2, col3 = st.columns([0.2, 5, 0.2])
        col2.image(image_query, use_column_width=True)

        print('Imagen cargada')
    
        # BUSCAR
        opciones = list(range(1, 11))
        with st.form("Buscar"):
            n_imagenes_similares = st.selectbox("Número de imágenes", opciones)
            boton = st.form_submit_button("Buscar")
        

        if boton:
            _, col2, _ = st.columns([0.2, 5, 0.2])
            texto = "Resultados de búsqueda"
            col2.markdown(f"<div style='text-align: center; font-weight: bold; font-size: larger;'>{texto}</div>", unsafe_allow_html=True) 
            
            # Preprocesar la imagen de consulta
            imagen_query_cargada = cargar_y_preprocesar_imagen(image_query)
            vgg19_query = st.session_state.modelo_vgg19.predict(imagen_query_cargada)
            vgg19_query = vgg19_query.flatten()

            # parámetros del modelo
            k_vecinos = 2
            metrica = 'cosine'

            # resultado de la búsqueda
            imagenes_similares, etiquetas_similares= buscar_imagenes_KNN(st.session_state.caracteristicas_vgg19, 
                                                        vgg19_query, 
                                                        k_vecinos, metrica, 
                                                        n_imagenes_similares)
            
            cols = st.columns(2)

            
            # Iterar sobre las imágenes y etiquetas y mostrarlas en las columnas
            for i, (imagen, etiqueta) in enumerate(zip(imagenes_similares, etiquetas_similares)):
                # Calcular la columna actual (0 o 1)
                col_index = i % 2

                # Mostrar la imagen y la etiqueta en la columna correspondiente
                cols[col_index].image(filtro_laplaciano(imagen), width=275, use_column_width=True, caption=etiqueta)
            print('Búsqueda completada')

if __name__ == '__main__':
    main()