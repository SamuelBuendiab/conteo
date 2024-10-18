import matplotlib.pyplot as plt
from PIL import Image

# Variables globales para almacenar los puntos
red_point = None
green_points = []

# Función para manejar los clics en la imagen
def onclick(event):
    global red_point, green_points

    # Verificamos si se hizo clic izquierdo o derecho
    if event.button == 1:  # Click izquierdo
        if len(green_points) < 2:
            # Agregar un nuevo punto verde
            green_points.append((event.xdata, event.ydata))
        else:
            # Si ya hay 2 puntos verdes, eliminar el último
            green_points.pop(0)  # Eliminar el primer punto
            green_points.append((event.xdata, event.ydata))
    
    elif event.button == 3:  # Click derecho
        if red_point is None:
            # Agregar un nuevo punto rojo
            red_point = (event.xdata, event.ydata)
        else:
            # Si ya hay un punto rojo, eliminarlo
            red_point = None

    # Redibujar la imagen y los puntos
    redraw()

def redraw():
    ax.clear()  # Limpiar el eje
    ax.imshow(image)
    
    # Dibujar los puntos verdes
    for point in green_points:
        plt.scatter(point[0], point[1], color='green', s=100)
    
    # Dibujar el punto rojo si existe
    if red_point is not None:
        plt.scatter(red_point[0], red_point[1], color='red', s=100)
        # Dibujar líneas desde el punto rojo a cada punto verde
        for green_point in green_points:
            plt.plot([red_point[0], green_point[0]], [red_point[1], green_point[1]], color='blue')

    plt.axis('off')  # Ocultar ejes
    plt.draw()  # Redibujar la imagen

# Cargar la imagen
img_path = 'vert.png'  # Cambia esto por la ruta de tu imagen
image = Image.open(img_path)

# Mostrar la imagen
fig, ax = plt.subplots()
plt.axis('off')  # Ocultar ejes

# Conectar el evento de clic
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Mostrar la ventana
plt.show()