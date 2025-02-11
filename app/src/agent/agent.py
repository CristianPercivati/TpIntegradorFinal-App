from datetime import datetime
from agent.dqn import DQNAgent
from agent.dia import Dia
from agent.pasajero import Pasajero
import random
import numpy as np
import pandas as pd
from agent.utils import seasonality_normalized, historico, distribucion_clima, distribucion_zonas, distribucion_trafico, flota_posible, distribucion_zonas, generar_datos_2025, generar_pasajeros
import os

class Empresa:
    def __init__(self, W_CLIMA, W_TRAFICO, W_FERIADO, W_DEMANDA):
        self.fecha_simulacion = datetime(2023, 1, 1).date()
        self.tarifa_base = 3
        self.prob_descuento = 0.2
        self.compensar_peajes = 0.5
        self.vehiculos = ['Minivan']
        self.zonas_activas = [3]
        self.agent = DQNAgent(state_size=8, action_size=12)  # 12 tipos de acciones
        #Traemos los pesos que definimos antes
        self.w_clima = W_CLIMA
        self.w_trafico = W_TRAFICO
        self.w_feriado = W_FERIADO
        self.w_demanda = W_DEMANDA

    def acciones(self):
      #Definimos qué tipo de acciones puede tomar el agente, y un valor param que dice cuánto puede subir o bajar de esa acción.
        return [
            {'tipo': 'aumentar_tarifa', 'param': 0.005},
            {'tipo': 'bajar_tarifa', 'param': 0.005},
            {'tipo': 'aumentar_descuento', 'param': 0.1},
            {'tipo': 'bajar_descuento', 'param': 0.1},
            {'tipo': 'agregar_vehiculo', 'param': "Camioneta"},
            {'tipo': 'agregar_zona', 'param': 1},
            {'tipo': 'elevar_peso_clima', 'param': 0.05},
            {'tipo': 'elevar_peso_feriado', 'param': 0.05},
            {'tipo': 'bajar_peso_clima', 'param': 0.05},
            {'tipo': 'bajar_peso_feriado', 'param': 0.05},
            {'tipo': 'subir_peso_demanda', 'param': 0.05},
            {'tipo': 'bajar_peso_demanda', 'param': 0.05},
        ]
    #Lo que hace esta función es "traducir" las acciones generadas por el agente para que modifique su estado.
    def aplicar_accion(self, action_idx):
        acc = self.acciones()[action_idx]
        if acc['tipo'] == 'aumentar_tarifa':
            self.tarifa_base *= (1 + acc['param'])
        elif acc['tipo'] == 'bajar_tarifa':
            self.tarifa_base *= (1 - acc['param'])
        elif acc['tipo'] == 'aumentar_descuento':
            self.prob_descuento = min(1.0, self.prob_descuento + acc['param'])
        elif acc['tipo'] == 'bajar_descuento':
            self.prob_descuento = max(0.0, self.prob_descuento - acc['param'])

        #Lo único diferente es esto, la forma en que agregamos nuevas zonas y vehículos.
        elif acc['tipo'] == 'agregar_vehiculo' and acc['param'] not in self.vehiculos:
            vehiculo_nuevo = random.choice(list(flota_posible.keys()))
            if vehiculo_nuevo not in self.vehiculos:
                self.vehiculos.append(vehiculo_nuevo)
        elif acc['tipo'] == 'agregar_zona':
            zona_random = random.choice(list(distribucion_zonas.keys()))
            if zona_random not in self.zonas_activas and len(self.zonas_activas) < len(distribucion_zonas):
                self.zonas_activas.append(zona_random)

        elif acc['tipo'] == 'elevar_peso_clima' and self.w_clima<0.35:
            self.w_clima = self.w_clima + acc['param']
        elif acc['tipo'] == 'elevar_peso_feriado' and self.w_feriado<0.35:
            self.w_feriado = self.w_feriado + acc['param']
        elif acc['tipo'] == 'elevar_peso_demanda' and self.w_demanda<1.5:
            self.w_demanda = self.w_demanda + acc['param']
        elif acc['tipo'] == 'bajar_peso_demanda' and self.w_demanda>0:
            self.w_demanda = self.w_demanda - acc['param']
        elif acc['tipo'] == 'bajar_peso_clima' and self.w_clima>0:
            self.w_clima = self.w_clima - acc['param']
        elif acc['tipo'] == 'bajar_peso_feriado' and self.w_feriado>0:
            self.w_feriado = self.w_feriado - acc['param']

    def get_state(self, dia):
        return np.array([[
            self.tarifa_base / 200,
            self.prob_descuento,
            self.compensar_peajes,
            len(self.vehiculos)/5,
            len(self.zonas_activas)/15,
            distribucion_clima[dia.clima],
            dia.trafico_level/2,
            1 if dia.es_feriado else 0
        ]])


    def calcular_tarifa_final(self, dia, zona, costo_minimo):
        ajuste_clima = 1 + (self.w_clima * distribucion_clima[dia.clima])
        ajuste_trafico = 1 + (self.w_trafico * dia.trafico_level * 0.75)
        ajuste_feriado = 1 + (self.w_feriado * dia.es_feriado * 0.2)
        ajuste_demanda = 1 + (self.w_demanda * dia.ratio_estacionalidad * 0.2)

        tarifa = self.tarifa_base * ajuste_clima * ajuste_trafico * ajuste_feriado * ajuste_demanda
        if random.random() < self.prob_descuento:
            tarifa *= 0.9
        if random.random() < self.compensar_peajes:
            tarifa -= distribucion_zonas[zona][3] * 5
        costo_minimo = costo_minimo *1.2 # Margen del 20%
        return max(tarifa, costo_minimo)  # Asegurar precio sobre costo

    def calcular_costos(self, dia, vehiculo, zona):
        costo = distribucion_zonas[zona][2] #Costos estimados por zona, un 25% de la media de tarifas
        costos_vehiculo = costo * flota_posible[vehiculo][1] #Costo por zona * el costo extra del vehículo
        costo *= 1 + (dia.trafico_level * 0.25) #Un peso de 0.25 de importancia por nivel de tráfico
        return costo * costos_vehiculo

    def train(self, model_name, episodes=1, days=5, passengers_per_day=30):
        batch_size = 1
        df_training = []
        paso_actual = 0
        total_pasos = episodes * days
        for e in range(episodes):
            total_reward = 0
            #Reiniciar fecha cada episodio
            self.fecha_simulacion = datetime(2023, 1, 1).date()
            for d in range(days):
                total_costo=0
                total_tarifa=0
                dia = Dia(self.fecha_simulacion)
                self.fecha_simulacion = dia.fecha_actual
                state = self.get_state(dia)
                action_idx = self.agent.act(state)
                self.aplicar_accion(action_idx)

                pasajeros_aceptados = 0
                for _ in range(round(passengers_per_day*dia.ratio_estacionalidad)):
                    aceptado = False
                    p = Pasajero(self.zonas_activas, dia)
                    costo = self.calcular_costos(dia, p.vehiculo, p.zona)
                    tarifa = self.calcular_tarifa_final(dia, p.zona,costo)
                    if p.tomar_viaje(tarifa, self.zonas_activas, 0):
                        aceptado = True
                        pasajeros_aceptados += 1
                        total_costo=total_costo+costo
                        total_tarifa=total_tarifa+tarifa
                    #Agregamos el viaje que se pidió, ya sea aceptado o no, a una lista.
                    df_training.append(generar_datos_2025(p, dia, dia.fecha_actual, tarifa, p.vehiculo, aceptado))

                #Calcular recompensa (beneficio neto)
                vehiculo = random.choice(self.vehiculos)
                reward = (total_tarifa - total_costo) * pasajeros_aceptados
                total_reward += reward
                next_state = self.get_state(dia)
                self.agent.remember(state, action_idx, reward, next_state, False)

                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)
                paso_actual = paso_actual+1
                progress = (paso_actual/total_pasos)*100
                yield {
                    "progress": progress,
                    "episode": e+1,
                    "day": d+1,
                    "accepted": pasajeros_aceptados,
                    "reward": reward,
                }
                print(f'Episodio: {e+1}, Día: {d+1}, Pasajeros: {pasajeros_aceptados}, Recompensa: {reward:.2f}')
                print(passengers_per_day*dia.ratio_estacionalidad)

            print(f'Episodio {e+1} finalizado. Recompensa total: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.3f}')
        #Una vez finalizado el entrenamiento, convertimos la lista con los resultados del entrenamiento a un dataframe.
        df_training_done = pd.DataFrame(df_training)
        columnas_originales = [
                'Fecha', 'Pasajero', 'Costo', 'Peajes', 'Espera', 'Parking', 'Extra', 'Total',
                'cantidad_pasajeros', 'Tipo_Servicio', 'Ubicación', 'venta', 'fecha', 'total_usd',
                'clima', 'temperatura', 'autopista', 'trafico', 'feriado', 'genero', 'Rating', 'Review', "Aceptado"
            ]
        df_training_done = df_training_done[columnas_originales]
        df_training_done.to_csv(f'serving\\outputs\\{model_name}_output.csv', index=False)
        #Salvar el modelo:
        fecha = datetime.now().strftime("%Y%m%d")
        if not os.path.exists(f'serving\\models\\{model_name}'):
            os.makedirs(f'serving\\models\\{model_name}')
        self.agent.model.save(f'serving\\models\\{model_name}\\{fecha}_model.keras')
        yield {
            "progress": 100
        }