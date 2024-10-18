import cv2
import numpy as np

def calculate_orientation_field(image, block_size):
    """
    Calcula el campo de orientación de las crestas en la imagen.
    """
    rows, cols = image.shape
    orientation_field = np.zeros((rows, cols), dtype=np.float32)

    # Dividir la imagen en bloques pequeños y calcular el ángulo de las crestas en cada bloque
    for y in range(0, rows, block_size):
        for x in range(0, cols, block_size):
            block = image[y:y + block_size, x:x + block_size]

            # Gradientes Sobel
            grad_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)

            # Promedio de los gradientes en cada bloque para calcular la orientación dominante
            Gxx = np.mean(grad_x**2)
            Gyy = np.mean(grad_y**2)
            Gxy = np.mean(grad_x * grad_y)

            # Calcular el ángulo de las crestas (en radianes)
            angle = 0.5 * np.arctan2(2 * Gxy, Gxx - Gyy)
            orientation_field[y:y + block_size, x:x + block_size] = angle

    return orientation_field

def draw_orientation_field(image, orientation_field, block_size):
    """
    Dibuja el campo de orientación en la imagen original.
    """
    rows, cols = image.shape
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for y in range(0, rows, block_size):
        for x in range(0, cols, block_size):
            angle = orientation_field[y, x]
            x_end = int(x + block_size * np.cos(angle))
            y_end = int(y + block_size * np.sin(angle))
            cv2.line(output, (x, y), (x_end, y_end), (0, 255, 0), 1)

    return output

def detect_nucleus(orientation_field, block_size):
    """
    Detecta el núcleo basado en la variación de la orientación de las crestas.
    """
    rows, cols = orientation_field.shape
    max_variation = 0
    nucleus_position = (0, 0)

    # Recorremos la imagen en bloques para calcular la variación de ángulos
    for y in range(block_size, rows - block_size, block_size):
        for x in range(block_size, cols - block_size, block_size):
            # Tomar los ángulos de los bloques vecinos
            angle = orientation_field[y, x]
            neighbors = orientation_field[y-block_size:y+block_size, x-block_size:x+block_size]

            # Calcular la variación de ángulos (diferencia estándar)
            angle_variation = np.std(neighbors)

            # Si la variación es mayor que el máximo encontrado, actualizar el núcleo
            if angle_variation > max_variation:
                max_variation = angle_variation
                nucleus_position = (x, y)

    return nucleus_position

# Cargar la imagen de la huella dactilar en escala de grises
image = cv2.imread('verticilo\image7.png', cv2.IMREAD_GRAYSCALE)

# Invertir la imagen si las crestas están en blanco y el fondo en negro
inverted_image = cv2.bitwise_not(image)

# Aplicar suavizado para reducir el ruido
blurred = cv2.GaussianBlur(inverted_image, (5, 5), 0)

# Calcular el campo de orientación de las crestas
block_size = 16  # Tamaño del bloque para analizar la orientación local
orientation_field = calculate_orientation_field(blurred, block_size)

# Detectar el núcleo basado en la variación de la orientación
nucleus_position = detect_nucleus(orientation_field, block_size)

# Dibujar el campo de orientación sobre la imagen original
output = draw_orientation_field(image, orientation_field, block_size)

# Marcar el núcleo detectado con un punto rojo
cv2.circle(output, nucleus_position, 5, (0, 0, 255), -1)
cv2.putText(output, "Nucleo", (nucleus_position[0] - 20, nucleus_position[1] - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Mostrar la imagen con el núcleo detectado
cv2.imshow('Nucleo Detectado', output)
cv2.waitKey(0)
cv2.destroyAllWindows()



