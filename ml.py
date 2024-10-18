import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from glob import glob

# Definir tamaño de las imágenes (ajustar según tus datos)
img_size = 128

# Función para procesar el archivo XML y extraer coordenadas de puntos clave
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = root.findall('object')
    keypoints = []

    for obj in objects:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        x_center = (int(bndbox.find('xmin').text) + int(bndbox.find('xmax').text)) / 2
        y_center = (int(bndbox.find('ymin').text) + int(bndbox.find('ymax').text)) / 2
        keypoints.append((x_center, y_center))  # Guardamos coordenadas

    return keypoints

# Función para cargar imágenes y sus respectivas anotaciones
def create_training_data(images_folder, xml_folder):
    X_train = []
    y_train = []

    for xml_file in glob(os.path.join(xml_folder, "*.xml")):
        keypoints = parse_xml(xml_file)

        # Obtener el nombre del archivo de imagen
        filename = os.path.splitext(os.path.basename(xml_file))[0] + '.png'  # Ajusta según la extensión de tus imágenes
        image_path = os.path.join(images_folder, filename)

        # Cargar la imagen y redimensionar
        image = cv2.imread(image_path)
        if image is None:
            print(f"Imagen no encontrada: {image_path}")
            continue

        image = cv2.resize(image, (img_size, img_size))
        X_train.append(image)

        # Aplanar keypoints y asegurarte de que haya 6 puntos (3 puntos clave * 2 coordenadas)
        if len(keypoints) == 3:  # Asegurarse de que haya exactamente 3 puntos clave
            flat_keypoints = [coord for point in keypoints for coord in point]  # Aplanar las coordenadas
            y_train.append(flat_keypoints)
        else:
            print(f"Advertencia: El archivo XML {xml_file} no tiene 3 puntos clave (tiene {len(keypoints)}).")

    # Convertir a un arreglo NumPy
    return np.array(X_train), np.array(y_train)

# Ruta a las carpetas con las imágenes y los archivos XML
images_folder = "imagenes\images"
xml_folder = "imagenes\Annotations"

# Crear los datos de entrenamiento
X_train, y_train = create_training_data(images_folder, xml_folder)

# Sincronizar X_train y y_train
X_train = np.array(X_train)
y_train = np.array(y_train)

# Asegúrate de que las longitudes coincidan
if len(X_train) != len(y_train):
    # Sincroniza eliminando las imágenes que no tienen coordenadas
    min_length = min(len(X_train), len(y_train))
    X_train = X_train[:min_length]
    y_train = y_train[:min_length]

print("Después de la sincronización:")
print("Número de imágenes:", len(X_train))
print("Número de etiquetas:", len(y_train))

# Verificar tamaños
print("Número de imágenes:", len(X_train))
print("Número de etiquetas:", len(y_train))

# Asegúrate de que los tamaños sean iguales
if len(X_train) != len(y_train):
    raise ValueError(f"Error: Las longitudes no coinciden. X_train tiene {len(X_train)} muestras y y_train tiene {len(y_train)} muestras.")

# Normalizar imágenes
X_train = X_train / 255.0  # Normalizar las imágenes a [0,1]

# Definir el modelo de CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6)  # 6 coordenadas (3 puntos clave * 2 coordenadas x, y)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Guardar el modelo entrenado
model.save('modelo_puntos_clave.keras')

print("Entrenamiento completado y modelo guardado.")




