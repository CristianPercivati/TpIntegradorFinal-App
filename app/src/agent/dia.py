from datetime import datetime, timedelta
import numpy as np
from agent.utils import seasonality_normalized, historico

class Dia:
    def __init__(self, fecha_inicio = None):
        self.fecha_actual = fecha_inicio if fecha_inicio else datetime(2023, 1, 1).date()
        # Buscar en histórico
        if self.fecha_actual in historico:
            registro = historico[self.fecha_actual]
            self.clima = registro['clima'] if registro['clima'] not in ["Tormenta eléctrica"] else 'Tormenta'
            self.trafico_level = self._mapear_trafico(registro['trafico'])
            self.es_feriado = registro['feriado']
            self.ratio_estacionalidad = seasonality_normalized[self.fecha_actual]
        else:
            # Generar aleatorio si no hay datos
            self.clima = np.random.choice(["Despejado", "Nublado", "Llovizna", "Tormenta eléctrica", "Niebla"])
            self.trafico_level = np.random.choice([0, 1, 2])
            self.es_feriado = np.random.choice([False, True], p=[0.9, 0.1])
            self.ratio_estacionalidad = np.random.uniform(0.5, 1.5)

        # Avanzar fecha para próximo día
        self.fecha_actual += timedelta(days=1)

    def _mapear_trafico(self, trafico_str):
        mapeo = {'Bajo': 0, 'Medio': 1, 'Alto': 2}
        return mapeo.get(trafico_str, np.random.choice([0, 1, 2]))

    def __repr__(self):
        return f"Dia({self.fecha_actual - timedelta(days=1)}): {self.clima}, Tráfico: {['Bajo','Medio','Alto'][self.trafico_level]}, Feriado: {self.es_feriado}"
    
    