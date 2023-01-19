import os
import sys
import argparse
import simple_air_hockey
import time
from datetime import datetime
import pytz
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque
import pickle
import warnings
import time
import evaluate_policy
warnings.filterwarnings("ignore")
America = pytz.timezone("America/New_York")


# Implementación de TD3 (Twin DDPG)
# La codificación aquí mostrada es el resultado de culminar el curso https://www.udemy.com/course/aprendizaje-por-refuerzo-profundo/
# El cual explora los detalles de TD3 y cuya referencia oficial es el paper https://arxiv.org/abs/1802.09477
# y el repositorio oficial de TD3 https:// github.com/sfujim/TD3


# DEFINIENDO LAS ARQUITECTURAS A USAR
# ----------------------------------------
# En primer lugar definimos nuestro almacen de transiciones, debemos entender que una transicion
# es una cuaterna cuyos elementos son: 
# Estado Actual(S_t), Acción ejecutada (a_t), Recompensa Obtenida (r_t), Estado Siguiente(s_(t+1)) y done de tipo bool
# Este almacen es llamado Replay Buffer y debe tener dos funciones: Guardado de transición y Selección Aleatoria

class Replay_Buffer(object):

    def __init__(self,capacidad=1e6,scara="right"):
        self.transiciones = []
        self.transiciones_totales = 0
        self.capacidad = capacidad
    
    def guarda_transicion(self, transicion):
        # Si la capacidad esta agotada elimina la transicion mas antigua
        # y sigue llenando con un transición nueva
        if self.transiciones_totales == self.capacidad:
           self.transiciones.pop(0)
        else:
           self.transiciones_totales += 1
        self.transiciones.append(transicion)

    def seleccion_muestra(self, numero_elementos):
        muestra = []
        indices = np.random.choice(self.transiciones_totales,numero_elementos)
        for i in indices:
            muestra.append(self.transiciones[i])
        return muestra

    def respaldar(self,scara):
        np.save("./replaybuffer/replay_buffer_"+str(scara)+".npy", self.transiciones)
        print("Replay Buffer Guardado. Tamaño de Replay Buffer: ",len(self.transiciones_totales))
        now = datetime.now(America)
        print("Fecha: ",now.date(),"Hora: ",now.time())

    def cargar(self,scara):
        self.transiciones = np.load("./replaybuffer/replay_buffer_"+str(scara)+".npy",allow_pickle=True).tolist()
        self.transiciones_totales = len(transiciones)
        print("Replay Buffer Cargado: Tamaño de Replay Buffer: ",self.transiciones_totales)

# En segundo lugar crearemos nuestras redes neuronales, debemos heredar los metodos de nn.Module
# que permitirá llamar a metodos tales como nn.Linear, funciones de activación, etc.

class Actor(nn.Module):

    def __init__(self, dimension_estados, dimension_acciones):
        super(Actor, self).__init__()
        self.capa_entrada = nn.Linear(dimension_estados,400)
        self.capa_oculta = nn.Linear(400,300)   
        self.capa_salida = nn.Linear(300,dimension_acciones)

    def forward(self, estado):
        x = F.relu(self.capa_entrada(estado))
        x = F.relu(self.capa_oculta(x))
        accion = torch.tanh(self.capa_salida(x))
        return accion

# En tercer lugar definiremos una clase que involucre a dos criticos gemelos:

class Criticos_Gemelos(nn.Module):

    def __init__(self, dimension_estados, dimension_acciones):
        super(Criticos_Gemelos, self).__init__()
        self.capa_entrada_critico1 = nn.Linear(dimension_estados + dimension_acciones,400)
        self.capa_oculta_critico1 = nn.Linear(400,300)   
        self.capa_salida_critico1 = nn.Linear(300,1) # La salida es de dimension 1 debido a que se predice el Q-value

        self.capa_entrada_critico2 = nn.Linear(dimension_estados + dimension_acciones,400)
        self.capa_oculta_critico2 = nn.Linear(400,300)   
        self.capa_salida_critico2 = nn.Linear(300,1) # La salida es de dimension 1 debido a que se predice el Q-value

    def forward(self, estado, accion): 

        # La entrada es un par estado, acción
        par_estado_accion = torch.cat([estado,accion],1) # Concatena de forma vertical
        
        #Uniremos ambos fordward de los dos gemelos, debido a que siempre haremos estos calculos en simultaneo.
        x_critico1 = F.relu(self.capa_entrada_critico1(par_estado_accion))
        x_critico1 = F.relu(self.capa_oculta_critico1(x_critico1))
        q_value_critico1 = self.capa_salida_critico1(x_critico1)

        x_critico2 = F.relu(self.capa_entrada_critico2(par_estado_accion))
        x_critico2 = F.relu(self.capa_oculta_critico2(x_critico2))
        q_value_critico2 = self.capa_salida_critico2(x_critico2)
        
        # La salida es el Q-value
        return q_value_critico1,q_value_critico2


class TD3(object):
    def __init__(self, scara, dimension_estados, dimension_acciones, ratio_aprendizaje = 0.0005, tamano_batch = 100):

        self.scara = scara
        self.replay_buffer = Replay_Buffer(scara=scara)
        self.perdida_critico1,self.perdida_critico2,self.perdida_actor = [],[],[]

        # Instanciamos los objetos de redes neuronales para los actores y críticos:
        self.actor = Actor(dimension_estados, dimension_acciones)
        self.actor_target = Actor(dimension_estados, dimension_acciones)
        self.criticos_gemelos = Criticos_Gemelos(dimension_estados, dimension_acciones)
        self.criticos_gemelos_targets = Criticos_Gemelos(dimension_estados, dimension_acciones)

        # Al inicio se hará una copia literal de los pesos del actor y críticos hacia el actor target y críticos target
        # respectivamente
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.criticos_gemelos_targets.load_state_dict(self.criticos_gemelos.state_dict())

        # Se definirá además el optimizador que realizará el backpropagation en las redes, en este caso Adam.
        # Notar que no se incluyen a las redes targets debido a que estas no se actualizarán con bacpropagation
        # sino que se actualizaran mediante el promedio poliak que mas adelante se implementará.
        self.actor_optimizador = torch.optim.Adam(self.actor.parameters(),lr = ratio_aprendizaje)
        self.criticos_gemelos_optimizador = torch.optim.Adam(self.criticos_gemelos.parameters(),lr = ratio_aprendizaje)


    def accion(self, estado):
        # La accion que tomara el TD3 siempre será el ouput de la red Actor
        estado = torch.Tensor(estado.reshape(1, -1)) #Antes de poder enviar como entrada a la red debemos refomatearlo como 
                                                     # array unidimensional y luego convertirlo a Tensor
        return self.actor.forward(estado).data.numpy().flatten() #La salida la volvemos a convertir a array numpy para operar con comodidad
                                                # ↳ lo usaremos para aplanar a array unidimensional

    # A veces a medida que nos acerquemos a un optimo, requeriremos disminuir el tamaño del paso por lo que creamos
    # una utilidad para poder variar el ratio de aprendizaje:                                            
    def modificar_ratio_aprendizaje(self, alfa):
        print("Cambiando alfa (ratio de aprendizaje) a %f",alfa)
        self.actor_optimizador = torch.optim.Adam(self.actor.parameters(),lr = alfa)
        self.criticos_gemelos_optimizador  = torch.optim.Adam(self.criticos_gemelos.parameters(),lr = alfa)

    # SECCION DE ENTRENAMIENTO
    # ----------------------------------------
    def entrenamiento(self, gamma, numero_iteraciones,tamano_batch,ruido_politica,ruido_recorte_politica,freq_actualizacion, tau):
        
        for iteracion in range(numero_iteraciones):
            batch_estados_,bacth_acciones_,batch_recompensas_,batch_estados_siguientes_,batch_done_ = [],[],[],[],[]
            #print(iteracion)
            # # En primer lugar, empezaremos extrayendo transiciones del replay buffer de nuestras experiencias con el entorno.
            muestra = self.replay_buffer.seleccion_muestra(numero_elementos = tamano_batch)
            for elemento in muestra:
                batch_estados_.append(np.array(elemento[0],copy = False))
                bacth_acciones_.append(np.array(elemento[1], copy = False))
                batch_recompensas_.append(np.array(elemento[2], copy = False))
                batch_estados_siguientes_.append(np.array(elemento[3], copy = False))
                batch_done_.append(np.array(elemento[4], copy = False))
            # Convertimos a tensores, recordar que esto se usa para poder ingresarlo a la red neuronal.
            #print(np.array(batch_estados_))
            batch_estados = torch.Tensor(np.array(batch_estados_))
            batch_acciones = torch.Tensor(np.array(bacth_acciones_))
            batch_recompensas = torch.Tensor(np.array(batch_recompensas_).reshape(-1, 1))
            batch_estados_siguientes = torch.Tensor(np.array(batch_estados_siguientes_))
            batch_done = torch.Tensor(np.array(batch_done_).reshape(-1, 1))

            # Empezamos calculando un bloque de acciones siguientes a partir del bloque de estados siguientes:
            batch_acciones_siguientes = self.actor_target.forward(batch_estados_siguientes)

            #Creamos un lote de ruido gaussiano
            batch_ruido_gaussiano = torch.Tensor(bacth_acciones_).data.normal_(0, ruido_politica)
            batch_ruido_gaussiano.clamp(-ruido_recorte_politica,ruido_recorte_politica)
            batch_acciones_siguientes = (batch_acciones_siguientes + batch_ruido_gaussiano)
            # Recordar que las acciones de nuestro entorno solo reciben valores de -1 a 1 por lo que se deben recortar
            batch_acciones_siguientes = batch_acciones_siguientes.clamp(-1,1)

            # Ingresaremos los lotes de nuevos estados y nuevas acciones predichas y la ingresaremos a las redes de 
            # criticos targets para que nos arroje una estimación del Q-value de (s_(t+1),a_(t+1))
            Q_value_target_1,Q_value_target_2 = self.criticos_gemelos_targets.forward(batch_estados_siguientes,batch_acciones_siguientes)

            # Tomaremos el minimo de los dos valores Q obtenidos, recordemos que la finalidad es evitar la sobre-estimación
            min_Q_value_target = torch.min(Q_value_target_1,Q_value_target_2)

            # Calcularemos gracias a la relacion de Bellman para políticas deterministas el valor Q de 
            # un estado antescedente (Q(s_t,a_t)), ademas se multiplica por (1-batch_done) ya que recordemos que 
            # para estados terminales el Q_value del estado prescedente seria igual a la recompensa recibida en el estado terminal.
            Q_value_bellman =  batch_recompensas  + ((1-batch_done) * gamma * min_Q_value_target).detach()

            # Ahora se hará las predicciones de los criticos para poder calcular los Q values de los pares (s_t,a_t)
            Q_value_1,Q_value_2 = self.criticos_gemelos.forward(batch_estados,batch_acciones)

            # Ahora debemos calcular una perdida conjunta, partiendo de dos perdidas por separado que es el mean error square.
            # tratando de acercarnos cada vez mas a la prediccion Q_value_bellman
            perdida_critico_1 = F.mse_loss(Q_value_1,Q_value_bellman)
            perdida_critico_2 = F.mse_loss(Q_value_2,Q_value_bellman)
            perdida_conjunta_criticos = perdida_critico_1 + perdida_critico_2
            
            # Realizamos la propagacion hacia atras, actualizando los pesos de los dos criticos para reducir la perdida conjunta
            self.criticos_gemelos_optimizador.zero_grad()
            perdida_conjunta_criticos.backward()
            self.criticos_gemelos_optimizador.step()
            
            # Debemos ahora poder actualizar los pesos del actor con frecuencia de cada 2 iteraciones del entrenamiento, 
            # para ello necesitamos una funcion de desempeño, ya que se realizara un ascenso de gradiente.
            # La función de desempeño no es mas que el promedio de los valores Q obtenidos al evaluar el lote en el primer critico,
            # bien, como nuestro optimizador trabaja con una función de perdida, podemos simplemente tomar como perdida
            # al negativo de la función de desempeño, de esta manera al tratar de minimizar esta perdida logrará maximizar
            # la función de desempeño y en otras palabras se estará realizando un ascenso de gradiente.

            if iteracion % freq_actualizacion == 0:
                Q_value_1,Q_value_2 = self.criticos_gemelos.forward(batch_estados, self.actor(batch_estados))
                funcion_desempeno = Q_value_1.mean()   # Solo usaremos a Q_value_1
                perdida_actor = -funcion_desempeno
                # Realizamos la propagacion hacia atras logrando  maximizar la funcion de desempeño.
                
                self.actor_optimizador.zero_grad()
                perdida_actor.backward()
                self.actor_optimizador.step()
                #print(perdida_actor.detach().numpy())

            # Finalmente debemos actualizar nuestras redes targets mediante el promedio poliak. De igual manera será cada dos
            # veces, por lo que se incluirá dentro del if prescendente.
                for parametros,parametros_target in zip(self.actor.parameters(),self.actor_target.parameters()):
                    parametros_target.data.copy_(tau* parametros.data + (1-tau) * parametros_target.data)
                
                for parametros,parametros_target  in zip(self.criticos_gemelos.parameters(),self.criticos_gemelos_targets.parameters()):
                    parametros_target.data.copy_(tau* parametros.data + (1-tau) * parametros_target.data)
              
        self.perdida_critico1.append(perdida_critico_1)
        self.perdida_critico2.append(perdida_critico_2)
        self.perdida_actor.append(perdida_actor)

    def guardar_pesos_redes(self, scara):
        torch.save(self.actor.state_dict(), "models_TD3/TD3_actor_%s.pth" % (self.scara))
        torch.save(self.criticos_gemelos.state_dict(), "models_TD3/TD3_criticos_gemelos_%s.pth" % (self.scara))

    def cargar_pesos_redes(self, scara):
        self.actor.load_state_dict(torch.load("models_TD3/TD3_actor_%s.pth" % (self.scara)))
        self.criticos_gemelos.load_state_dict(torch.load("models_TD3/TD3_criticos_gemelos_%s.pth" % (self.scara)))


if __name__ == "__main__":

    # INSTANCIANDO LOS HIPERPARÁMETROS QUE USARÁN EN EL PROGRAMA:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semilla",                default = 0,       type=int   )  #  Semilla aleatoria del programa.
    parser.add_argument("--steps_exploracion",      default = 5e3,     type=int   )  #  Cantidad de steps con acciones aleatorias.
    parser.add_argument("--freq_evaluacion",        default = 2.5e3,   type=int   )  #  Cada cuanto se hace la evaluacion. 
    parser.add_argument("--steps_maximo",           default = 1e6,     type=int   )  #  Numero total de steps para entrenar.
    parser.add_argument("--guardar_modelos",        default = True,    type=bool  )  #  "True" Guarda los pesos de las redes.  
    parser.add_argument("--ruido_exploracion",      default = 0.1,     type=float )  #  Desviacion std de Ruido añadido a la salida del actor
    parser.add_argument("--tamano_batch",           default = 100,     type=int   )  #  tamaño de lote.
    parser.add_argument("--descuento_gamma",        default = 0.9999,  type=float )  #  Es el descuento aplicado a las recompensas.
    parser.add_argument("--tau",                    default = 0.005,   type=float )  #  Hiperparámetro para el promedio poliak.
    parser.add_argument("--ruido_politica",         default = 0.2,     type=float )  #  Desviacion std agregado a la salida del actor target.
    parser.add_argument("--ruido_recorte_politica", default = 0.5,     type=float )  #  Recorta a ruido_politica
    parser.add_argument("--freq_actualizacion",     default = 2,       type=int   )  #  
    parser.add_argument("--ratio_aprendizaje",      default = 0.0005,  type=float )  # 
    parser.add_argument("--scara",                  default = "right"             )  # 
    parser.add_argument("--cargar_modelos",         default = False,   type=bool  )  # 
    args = parser.parse_args()

    # INSTANCIANDO EL ENTORNO QUE SE CREO: "SimpleAirHockey-v0"
    # -------------------------------------------------------
    env = gym.make("SimpleAirHockey-v0")

    # Para que las experiencias de esta implementación sean repetibles en cualquier momento y en cualquier
    # hardware necesitamos declarar semillas aleatorias:
    env.seed(args.semilla)
    random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    np.random.seed(args.semilla)

    # Definimos algunas variables globales para poder trabajar con ellas:
    dimension_estados = env.observation_space.shape[0]
    dimension_acciones = env.action_space.shape[0]
    replay_buffer_episodio = []
    avg_reward_list = []
    ep_reward_list = []
    init = True
    avg_reward = 0
    steps_totales = 0
    timesteps_since_eval = 0
    cont = 0
    episode_num = 0
    done = True

    # INSTANCIANDO LA POLITICA TD3
    politica = TD3(scara = args.scara, dimension_estados = dimension_estados , dimension_acciones=dimension_acciones, 
    ratio_aprendizaje = args.ratio_aprendizaje, tamano_batch=args.tamano_batch)
    _ = evaluate_policy.evaluate(politica = politica, env = env, scara = args.scara)
    while steps_totales <= args.steps_maximo:

        if done:
            for transicion in replay_buffer_episodio:
                politica.replay_buffer.guarda_transicion(transicion)
            replay_buffer_episodio = []
            # Si no estamos en la primera de las iteraciones, arrancamos el proceso de entrenar el modelo
            if not init:
                if steps_totales > args.steps_exploracion and cont >= 2 and episode_reward>-200: #eliminando valores atípicos 
                    ep_reward_list.append(episode_reward)
                    avg_reward = np.mean(ep_reward_list[-100:])
                    avg_reward_list.append(avg_reward)
                print("Total Timesteps: {} Episode Timesteps: {} Episode Num: {} Disco coords: {} Reward: {:.3f} Avg.Reward: {:.3f}"
                .format(steps_totales,episode_timesteps, episode_num,env.coords(), episode_reward,avg_reward ))
                politica.entrenamiento(gamma=args.descuento_gamma, numero_iteraciones=episode_timesteps, tamano_batch=args.tamano_batch, ruido_politica=args.ruido_politica, 
                    ruido_recorte_politica= args.ruido_recorte_politica, freq_actualizacion=args.freq_actualizacion, tau=args.tau)
                
            # Evaluamos el episodio y guardamos la política si han pasado las iteraciones necesarias
            if timesteps_since_eval >= args.freq_evaluacion:
                cont = cont +1
                timesteps_since_eval %= args.freq_evaluacion
                _ = evaluate_policy.evaluate(politica = politica, env = env, scara = args.scara)
                politica.guardar_pesos_redes(scara=args.scara)
                pickle.dump(avg_reward_list, file = open("./results/avg_reward_"+str(args.scara)+".pkl", "wb"))
                pickle.dump(ep_reward_list, file = open("./results/ep_reward_"+str(args.scara)+".pkl", "wb"))
                pickle.dump(steps_totales, file = open("./results/total_timesteps_"+str(args.scara)+".pkl", "wb"))
                pickle.dump(episode_num, file = open("./results/episode_num_"+str(args.scara)+".pkl", "wb"))
                pickle.dump(politica.perdida_critico1, file = open("./results/critico1_loss_"+str(args.scara)+".pkl", "wb"))
                pickle.dump(politica.perdida_critico2, file = open("./results/critico2_loss_"+str(args.scara)+".pkl", "wb"))
                pickle.dump(politica.perdida_actor, file = open("./results/actor_loss_"+str(args.scara)+".pkl", "wb"))

                
            # Cuando el entrenamiento de un episodio finaliza, reseteamos el entorno
            obs = env.reset(scara=args.scara)
            
            # Configuramos el valor de done a False
            done = False
            
            # Configuramos la recompensa y el timestep del episodio a cero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            #if avg_reward > 400 :
                #politica.modificar_ratio_aprendizaje(0.0001)
            
        # Al inicio realizamos una exploracion con acciones aleatorias con la finalidad de ingresar transiciones al replay buffer
        # y el metodo de entrenamiento tenga un lote de transiciones con el cual trabajar.
        if steps_totales < args.steps_exploracion:
            accion = env.action_space.sample()
        
        else: # Sino Cambiamos al modelo
            accion = politica.accion(np.array(obs))
            # Si el valor de explore_noise no es 0, añadimos ruido a la acción y lo recortamos en el rango adecuado
            if args.ruido_exploracion != 0:
                accion = (accion + np.random.normal(0, args.ruido_exploracion, size = env.action_space.shape[0])).clip(-1.0, 1.0)
        
        # El agente ejecuta una acción en el entorno y alcanza el siguiente estado y una recompensa
    
        new_obs, reward, done, _ = env.step([accion,args.scara])
        
        # Comprobamos si el episodio ha terminado
        done_bool = 0 if episode_timesteps + 1 == env.max_steps() else float(done)
        
        # Incrementamos la recompensa total
        episode_reward += reward
        
        # Modificacion de la recompensa de transiciones (Estrategia de regreso en el tiempo)
        if env.step_colision_linea_enemiga != 0:
            transicion = list(replay_buffer_episodio[env.step_golpe_disco-1])
            transicion[2] = transicion[2] + reward
            replay_buffer_episodio[env.step_golpe_disco-1] = tuple(transicion)
            env.reset_steps_linea()
            reward = 0
        
        if env.step_anotacion != 0:
            transicion = list(replay_buffer_episodio[env.step_golpe_disco-1])
            transicion[2] = transicion[2] + reward
            replay_buffer_episodio[env.step_golpe_disco-1] = tuple(transicion)
            env.reset_step_anotacion()
            reward = 0
        
        replay_buffer_episodio.append((obs, accion, reward, new_obs ,done_bool))
        
        # Actualizamos el estado, el timestep del número de episodio, el total de timesteps y el número de pasos desde la última evaluación de la política
        obs = new_obs
        episode_timesteps += 1
        steps_totales += 1
        init = False
        timesteps_since_eval += 1
    
