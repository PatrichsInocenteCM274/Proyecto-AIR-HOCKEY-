

# ✨ Proyecto AIR HOCKEY en PyBullet ✨

Este proyecto tuvo como objetivo crear un entorno de Gimnasio en pybullet desde cero con la finalidad de entrenar dos agentes y enseñarles a jugar Air Hockey, los agentes robóticos son scaras de 2 grados de libertad que poseen un mazo en su efector final. 
Puede usted ver una demostración completa dando **Ctrl + click** al siguiente icono: 

[![Abrir video en Youtube](https://badgen.net/badge/Proyecto/Youtube/red?)](https://www.youtube.com/watch?v=1iVLO5VqGB4&list=PLZ8tLoVVysdPebF_fTWDxCnU2IsV_-gIr&index=1)

El aprendizaje de los agentes se ha logrado mediante el Modelo TD3 (Twin delayed DDPG) cuya codificación ha sido implementada como parte de la culminación del curso https://www.udemy.com/course/aprendizaje-por-refuerzo-profundo/ cuyas referencias oficiales son: 
1. El repositorio oficial del equipo creador del Modelo TD3 https://github.com/sfujim/TD3
2. El Articulo oficial https://arxiv.org/abs/1802.09477. 

Además de ello, se ha hecho una comparativa de desempeño en entrenamiento de TD3 versus DDPG (Articulo https://arxiv.org/abs/1509.02971), podrá observarlo al final de este documento.



https://user-images.githubusercontent.com/30361234/208005553-9cb6a1f3-f142-403e-b957-4182b9c3076e.mp4

### <ins>Analisis de Entorno:</ins>
En este proyecto se modelo un Entorno parcialmente observable por los agentes, donde en cada instante de tiempo, los agentes podian solo registrar los datos observación mostrados en la siguiente imagen, además se adjunta las acciones que pueden ejecutar los agentes. 

![observacion(1)](https://user-images.githubusercontent.com/30361234/221737585-acf74b14-bf5e-4b0c-af71-4deba00881d2.png)


### <ins>Controles en Pybullet:</ins>
Es importante que al momento de interactuar con el entorno pybullet, podamos rotar, desplazarnos y manipular los objetos, por ello adjunto una pequeña guía de los mas importantes controles de teclado y mouse disponible:

- **Scroll Mouse hacia abajo/arriba:** Zoom In/out de la escena.
- **Ctrl + keep Click + Desplazar mouse:** Rotación de la escena.
- **Ctrl + Mouse Left + Mouse Right + Desplazar mouse:** Movimiento paralelo en la escena.
- **Tecla G:** Retira o muestra las ventanas de Depuración.
- **Tecla Q:** Cambia a vista de Objetos de Colisión.
- **Keep Click + Desplazar mouse:** Manipula y mueve objetos.


## Instrucciones de Uso:
 
### <ins>EN COLAB</ins>
Los procesos de entrenamiento y procesamiento de videos se harán en colab para no repercutir en gastos computacionales de nuestra maquina local, por lo que se ha preparado un cuaderno, el cual puede usted acceder dando **Ctrl + click** en el siguiente icono:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrichsInocenteCM274/Proyecto-AIR-HOCKEY-/blob/master/Entorno_Air_Hockey_Entrenamiento_con_TD3_y_DDPG.ipynb) 
##### Los comandos que se muestran en el cuaderno son los siguientes:


##### 0. Para poder clonar el proyecto:  
~~~
!git clone https://github.com/PatrichsInocenteCM274/Proyecto-AIR-HOCKEY-.git 
%cd Proyecto-AIR-HOCKEY-/
!pip install -e .
~~~

##### 1. Entrenamiento de cada Scara desde cero (Se borrará el modelo entrenado anteriormente):  
Para entrenar la scara derecha con TD3:  
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

##### 3. Finalmente para descargar proyecto ejecutar:
~~~
from google.colab import files
!zip -r Proyecto-AIR-HOCKEY-.zip ./ -x ".git/*" "simple_air_hockey.egg-info/*" "**__pycache__/*"
files.download('Proyecto-AIR-HOCKEY-.zip')
~~~

### <ins> EN MAQUINA LOCAL (PROBADO EN LINUX)</ins>

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


https://user-images.githubusercontent.com/30361234/218318463-1c5f1838-c617-43be-9351-ad9583dd83e8.mp4



##### 2. Para mostrar el juego entre TD3 Y DDPG jugando Air Hockey (Abrirá una pantalla GUI), escribir el siguiente comando en consola:  
~~~
python3 td3_inferencia.py --scara=all --models=2
~~~


https://user-images.githubusercontent.com/30361234/218318212-1a5c1840-b55a-405b-ae57-1bc66f56d6a3.mp4



##### 3. Para poder comprender como los datos trabajan internamente, se ha preparado un modo Demo que permite observar el tablero con identificadores de coordenadas (Se abrirá una pantalla GUI), escriba en consola:

~~~
python3 demo.py 
~~~

https://user-images.githubusercontent.com/30361234/208754050-1c7c75ca-9b21-420a-b6f8-21ddcff7ee80.mp4


## Métricas de Entrenamiento:

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
python3 visualiza.py --scara=left --models=2!
~~~

![comparativa](https://user-images.githubusercontent.com/30361234/213372824-b62a2649-a33c-4e90-b6e8-b071e260a026.png)



