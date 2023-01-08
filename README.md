# Proyecto AIR HOCKEY en PyBullet

Este proyecto tuvo como objetivo crear un entorno de Gimnasio en pybullet desde cero con la finalidad de entrenar dos agentes y enseñarles a jugar Air Hockey, los agentes robóticos son scaras de 2 grados de libertad que poseen un mazo en su efector final.  
El aprendizage de los agentes se ha logrado mediante el aprendizaje TD3 (Twin delayed DDPG) cuya codificación oficial e investigación son proveidos en el repositorio oficial del equipo creador del Modelo TD3 https://github.com/sfujim/TD3. Para los fines de este proyecto, la codificación de TD3 en python en la que me he basado es parte del curso de UDEMY "Aprendizaje por Refuerzo Profundo 2.0 en Python" --> https://www.udemy.com/course/aprendizaje-por-refuerzo-profundo/, el cual completé y recomiendo absolutamente.  
Además de ello, se ha hecho una comparativa de TD3 versus DDPG (Antecesor de TD3), podrá observarlo al final de este documento.



https://user-images.githubusercontent.com/30361234/208005553-9cb6a1f3-f142-403e-b957-4182b9c3076e.mp4

### Controles en Pybullet:
Es importante que al momento de interactuar con el entorno pybullet, podamos rotar, desplazarnos y manipular los objetos, por ello adjunto una pequeña guía de los mas importantes controles de teclado y mouse disponible:

- **Scroll Mouse hacia abajo/arriba:** Zoom In/out de la escena.
- **Ctrl + keep Click + Desplazar mouse:** Rotación de la escena.
- **Ctrl + Mouse Left + Mouse Right + Desplazar mouse:** Movimiento paralelo en la escena.
- **Tecla G:** Retira o muestra las ventanas de Depuración.
- **Keep Click + Desplazar mouse:** Manipula y mueve objetos.


### Instrucciones de Uso:

## <ins>EN COLAB</ins>

Los procesos de entrenamiento y visualización de videos se harán en colab para no repercutir en gastos computacionales de nuestra maquina local, por lo que usted debería ingresar a Colab y ejecutar los siguientes comandos:
~~~
!git clone https://github.com/PatrichsInocenteCM274/Proyecto-AIR-HOCKEY-.git 
%cd Proyecto-AIR-HOCKEY-/
!pip install -e .
~~~

##### 1. Entrenamiento de cada Scara desde cero (Se borrará el modelo entrenado anteriormente):  
Para entrenar la scara derecha con TD3, escribimos:  
~~~
!python3 td3_.py --scara=right  
~~~
Para entrenar la scara izquierda con TD3, escribimos:  
~~~
!python3 td3_.py --scara=left  
~~~
Para entrenar solo scara izquierda con modelo DDPG desde cero:  
~~~
!python3 ddpg_train_scara_left.py
~~~

##### 2. Para capturar grabaciones del entrenamiento de cada robot scara (los videos se almacenarán en la carpeta del proyecto), escribir:  
En el caso de querer obtener la grabación de scara derecha:  
~~~
!python3 td3_inferencia.py --scara=right  
~~~
En el caso de querer obtener la grabación de scara izquierda:  
~~~
!python3 td3_inferencia.py --scara=left  
~~~

##### 3. Finalmente para descargar proyecto entrenado a nuestra maquina local, escribir:
~~~
from google.colab import files
!zip -r /content/Proyecto-AIR-HOCKEY-.zip /content/Proyecto-AIR-HOCKEY-/
files.download('/content/Proyecto-AIR-HOCKEY-.zip')
~~~

## <ins> EN MAQUINA LOCAL (PROBADO EN LINUX)</ins>

De forma local podremos mostrar interactivamente el resultado del entrenamiento mediante el entorno de pyBullet, por lo que aperture una consola y escriba los siguientes comandos:

##### Si usted ha entrenado en colab el modelo, descomprima el proyecto descargado, acceda a él e instale las dependencias con el comando:
~~~
pip install -e .
~~~

##### Caso contrario, usted desea probar directamente el entorno con el entrenamiento por defecto, escriba:
~~~
git clone https://github.com/PatrichsInocenteCM274/Proyecto-AIR-HOCKEY-.git 
cd Proyecto-AIR-HOCKEY-/
pip install -e .
~~~

##### 1. Para mostrar el juego entre dos agentes TD3 jugando Air Hockey (Abrirá una pantalla GUI), escribir el siguiente comando en consola:  
~~~
python3 td3_inferencia.py --scara=all --models=1
~~~




https://user-images.githubusercontent.com/30361234/208745597-156dc360-4461-4b2e-8075-a0e5c7fbcb0b.mp4




##### 2. Para mostrar el juego entre TD3 Y DDPG jugando Air Hockey (Abrirá una pantalla GUI), escribir el siguiente comando en consola:  
~~~
python3 td3_inferencia.py --scara=all --models=2
~~~

https://user-images.githubusercontent.com/30361234/208436476-b9fe48ae-4d40-4fad-96a8-29592c0ea0a2.mp4

##### 3. Para poder comprender como los datos trabajan internamente, se ha preparado un modo Demo que permite observar el tablero con identificadores de coordenadas (Se abrirá una pantalla GUI), escriba en consola:

~~~
python3 demo.py 
~~~

https://user-images.githubusercontent.com/30361234/208754050-1c7c75ca-9b21-420a-b6f8-21ddcff7ee80.mp4


### Métricas de Entrenamiento:

##### 4. Para observar las métricas resultantes del entrenamiento, se ha preparado un gráfico para cada robot scara, donde se muestra el progreso en una ventana de 100 episodios, para ver escribir en consola:
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
![compara](https://user-images.githubusercontent.com/30361234/208805612-41396dce-e14d-4f76-8624-d60b6bfa4dcf.png)



