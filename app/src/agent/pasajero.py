from agent.utils import distribucion_zonas, flota_posible
import numpy as np

class Pasajero:
    def __init__(self, zonas_activas, dia):
        self.peso_feriado = 0.25 if dia.es_feriado else 0
        self.peso_clima = 0
        if dia.clima == "Tormenta eléctrica":
          self.peso_clima = 0.75
        elif dia.clima == "Lluvia":
          self.peso_clima = 0.35

        self.tolerancia_precio = np.random.lognormal(mean=0, sigma=0.3) * (1+self.peso_clima+self.peso_feriado)
        self.zona = np.random.choice(list(distribucion_zonas.keys()), p=[distribucion_zonas[z][0] for z in distribucion_zonas.keys()])
        self.vehiculo = np.random.choice(list(flota_posible.keys()), p=[flota_posible[z][0] for z in flota_posible.keys()])

    def tomar_viaje(self, tarifa_ofrecida, zonas_disponibles, demora):
        precio_max = distribucion_zonas[self.zona][1] * (1 + self.tolerancia_precio)
        return tarifa_ofrecida <= precio_max and self.zona in zonas_disponibles