import numpy as np
import pandas as pd
import random
from collections import deque
from tensorflow.keras import models, layers, optimizers
from datetime import datetime, timedelta
import ast

from agent.dia import Dia
from agent.pasajero import Pasajero
from agent.dqn import DQNAgent
from agent.agent import Empresa
from fastapi import HTTPException
import json
from agent.utils import seasonality_normalized, historico, distribucion_clima, distribucion_zonas, distribucion_trafico, flota_posible, distribucion_zonas

PASAJEROS_POR_DIA = 40
W_CLIMA = 0.15
W_TRAFICO = 0.25
W_FERIADO = 0.3
W_DEMANDA = 0.75

def train_model(model_name, episodes, days):
    episodes = int(episodes)
    days = int(days)
    print("asdfg")
    empresa = Empresa(W_CLIMA, W_TRAFICO, W_FERIADO, W_DEMANDA)
    df_2025 = empresa.train(model_name, episodes,days,PASAJEROS_POR_DIA)
    # Aquí se itera sobre cada actualización que emite el método train.
    for update in empresa.train(model_name, episodes, days, PASAJEROS_POR_DIA):
        # Convertimos el diccionario a JSON y agregamos salto de línea.
        yield json.dumps(update) + "\n"
