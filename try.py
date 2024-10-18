import cv2
import numpy as np
from skimage.filters import gabor
from skimage.morphology import skeletonize

# Cargar la imagen de huella dactilar en escala de grises
img = cv2.imread('verticilo/image4.png', cv2.IMREAD_GRAYSCALE)

# Preprocesamiento: suavizado y mejora del contraste
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
img_eq = cv2.equalizeHist(img_blur)

# Aplicar la transformada de Gabor para realzar los patrones orientados
def apply_gabor(img):
    frequencies = [0.1, 0.2, 0.3]  # Frecuencias de Gabor para detectar texturas
    filtered_images = []
    for frequency in frequencies:
        filt_real, filt_imag = gabor(img, frequency=frequency)
        filtered_images.append(np.sqrt(filt_real**2 + filt_imag**2))
    return np.mean(filtered_images, axis=0)

gabor_img = apply_gabor(img_eq)

# Detección de bordes para encontrar las crestas y valles
edges = cv2.Canny(np.uint8(gabor_img), 50, 150)

# Refinar la imagen usando esqueletización (para identificar patrones de vértices)
skeleton = skeletonize(edges / 255).astype(np.uint8)

# Detección de esquinas usando el detector de Harris
corners = cv2.cornerHarris(skeleton, 2, 3, 0.04)

# Umbral para filtrar esquinas débiles
corners_dilated = cv2.dilate(corners, None)
_, corners_thresh = cv2.threshold(corners_dilated, 0.01 * corners_dilated.max(), 255, 0)
corners_thresh = np.uint8(corners_thresh)

# Identificación de componentes conectados (núcleo y vértices)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corners_thresh)

# Dibujar los resultados sobre la imagen original
output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for centroid in centroids:
    cv2.circle(output_img, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)

# Mostrar resultados
cv2.imshow('Nucleo y Vertices Detectados', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

