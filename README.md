# Proyecto AIR HOCKEY en PyBullet

Este proyecto tuvo como objetivo crear un entorno de Gimnasio en pybullet desde cero con la finalidad de entrenar dos agentes y enseñarles a jugar Air Hockey, los agentes robóticos son scaras de 2 grados de libertad que poseen un mazo en su efector final.  
El aprendizage de los agentes se ha logrado mediante el aprendizaje TD3 (Twin delayed DDPG) cuya codificación oficial e investigación son proveidos en el repositorio oficial del equipo creador del Modelo TD3 https://github.com/sfujim/TD3 y cuya codificación en la que me he basado y modificado para los fines de este proyecto se puede encontrar en el curso de UDEMY APRENDIZAJE POR REFUERZO 2.0 y ser visualizado en el siguiente repositorio https://github.com/joanby/drl2.0



https://user-images.githubusercontent.com/30361234/208005553-9cb6a1f3-f142-403e-b957-4182b9c3076e.mp4



### Instrucciones de Uso:

##### 1. Clonar el repositorio y ubicarnos dentro del repositorio localmente.  
##### 2. De ser posible en un entorno virtual, cargar las dependencias del proyecto mediante el comando:
~~~
pip install -e .
~~~
##### 3.1. Para mostrar el juego entre dos agentes TD3 jugando Air Hockey (Abrirá una pantalla GUI), escribir el siguiente comando en consola:  
~~~
python3 td3_inferencia.py --scara=all --models=1
~~~




https://user-images.githubusercontent.com/30361234/208745597-156dc360-4461-4b2e-8075-a0e5c7fbcb0b.mp4




##### 3.2. Para mostrar el juego entre TD3 Y DDPG jugando Air Hockey (Abrirá una pantalla GUI), escribir el siguiente comando en consola:  
~~~
python3 td3_inferencia.py --scara=all --models=2
~~~

https://user-images.githubusercontent.com/30361234/208436476-b9fe48ae-4d40-4fad-96a8-29592c0ea0a2.mp4


##### 4. Para capturar grabaciones del entrenamiento de cada robot scara, escribir:  
En el caso de querer obtener la grabación de scara derecha:  
~~~
python3 td3_inferencia.py --scara=right  
~~~
En el caso de querer obtener la grabación de scara izquierda:  
~~~
python3 td3_inferencia.py --scara=left  
~~~
##### 5. Para entrenar cada scara desde cero:  
Para entrenar la scara derecha con TD3, escribimos:  
~~~
python3 td3_.py --scara=right  
~~~
Para entrenar la scara izquierda con TD3, escribimos:  
~~~
python3 td3_.py --scara=left  
~~~
Para entrenar solo scara izquierda con modelo DDPG desde cero:  
~~~
python3 ddpg_train_scara_left.py
~~~

### Modo demo:

![demo](https://user-images.githubusercontent.com/30361234/208746819-afee8982-7e1b-4e3d-9518-023f67dc32e1.png)

Podemos observar el tablero con identificadores de coordenadas (Se abrirá una pantalla GUI), para poder comprender como los datos trabajan internamente, escribiendo en consola:
~~~
python3 demo.py 
~~~


### Métricas de Entrenamiento:
Para observar el progreso del entrenamiento, se ha preparado un gráfico para cada robot scara, donde se muestra el progreso en una ventana de 100 episodios, para ver escribir en consola:
##### Para visualizar métrica de la scara derecha:
~~~
python3 visualiza.py --scara=right
~~~
##### Para visualizar métrica de la scara Izquierda:
~~~
python3 visualiza.py --scara=left
~~~
##### Para visualizar comparativa de TD3 vs DDPG en la scara Izquierda:
~~~
python3 visualiza.py --scara=left --models=2
~~~
