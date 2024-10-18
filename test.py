import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Tamaño de imagen (debe coincidir con el tamaño de entrada del modelo)
img_size = 128

# Ruta del modelo guardado
model_path = 'modelo_puntos_clave.keras'

# Cargar el modelo entrenado
model = tf.keras.models.load_model(model_path)

# Función para cargar y preprocesar la imagen
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Imagen no encontrada: {image_path}")
    
    original_image = image.copy()  # Guardamos una copia para mostrar
    image = cv2.resize(image, (img_size, img_size))  # Redimensionar al tamaño esperado
    image = image / 255.0  # Normalizar a [0,1]
    
    return image, original_image

# Función para hacer predicciones y visualizar resultados
def predict_and_show(image_path):
    # Cargar y preprocesar la imagen
    processed_image, original_image = load_and_preprocess_image(image_path)

    # Hacer la predicción
    prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]

    # Extraer las coordenadas de la predicción
    keypoints = [(prediction[i], prediction[i + 1]) for i in range(0, len(prediction), 2)]

    # Dibujar los puntos en la imagen original
    for (x, y) in keypoints:
        cv2.circle(original_image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)  # Puntos verdes

    # Mostrar la imagen con los puntos
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Predicción de Núcleo y Verticilos')
    plt.show()

# Ruta a la imagen de prueba (ajusta la ruta según sea necesario)
image_to_test = "img2.png"

# Ejecutar la función de predicción y visualización
predict_and_show(image_to_test)

