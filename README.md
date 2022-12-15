# Proyecto AIR HOCKEY

Este proyecto tuvo como objetivo crear un entorno pybullet desde cero, y usar dos agentes robóticos scaras de 2 grados de libertad como jugadores del juego de mesa AIR HOCKEY, este hazaña se logra mediante el aprendizaje TD3 (Twin delayed DDPG) cuya codificación oficial e investigación son proveidos en el repositorio oficial del equipo creador del Modelo TD3 https://github.com/sfujim/TD3 y cuya codificación en la que me he basado y modificado para los fines de este proyecto se puede encontrar en el curso de UDEMY APRENDIZAJE POR REFUERZO 2.0 y ser visualizado en el siguiente repositorio https://github.com/joanby/drl2.0

[![1.png](https://i.postimg.cc/L4ZMN7mT/1.png)](https://postimg.cc/8sD3CZJJ)

### Instrucciones de Uso:

##### 1. Clonar el repositorio y ubicarnos dentro del repositorio localmente.  
##### 2. De ser posible en un entorno virtual, cargar las dependencias del proyecto mediante el comando:
~~~
pip install -e .
~~~
##### 3. Para mostrar el juego entre dos agentes TD3 jugando Air Hockey (Abrirá una pantalla GUI), escribir el siguiente comando en consola:  
~~~
python3 td3_inferencia.py --scara=all  
~~~
##### 4. Para capturar grabaciones del entrenamiento de cada robot scara, escribir:  
En el caso de querer grabación de scara derecha:  
~~~
python3 td3_inferencia.py --scara=right  
~~~
En el caso de querer grabación de scara izquierda:  
~~~
python3 td3_inferencia.py --scara=left  
~~~
##### 5. Para entrenar cada scara desde cero:  
Para entrenar la scara derecha, escribimos:  
~~~
python3 td3_.py --scara=right  
~~~
Para entrenar la scara izquierda, escribimos:  
~~~
python3 td3_.py --scara=left  
~~~

### Modo demo:
[![tabler.png](https://i.postimg.cc/Gh6N9PVm/tabler.png)](https://postimg.cc/8jBw0vT2)
Podemos observar el tablero con identificadores de coordenadas (Se abrirá una pantalla GUI), para poder comprender como los datos trabajan internamente, escribiendo en consola:
~~~
python3 demo.py --scara=left  
~~~


### Métricas de Entrenamiento:
Para observar el progreso del entrenamiento, se ha preparado un gráfico para cada robot scara, donde se muestra el progreso en una ventana de 100 episodios, para ver escribir en consola:
##### Para ver métrica de la scara derecha:
~~~
python3 visualiza.py --scara=right
~~~
##### Para ver métrica de la scara Izquierda:
~~~
python3 visualiza.py --scara=left
~~~
