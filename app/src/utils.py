import numpy as np

# Diccionario de ubicaciones con coordenadas reales
ubicaciones_coords = {
    "Almagro": (-34.6114, -58.4370),
    "Recoleta": (-34.5887, -58.3933),
    "Retiro": (-34.5940, -58.3816),
    "La Boca": (-34.6350, -58.3640),
    "Hipódromo Palermo": (-34.5667, -58.4364),
    "Aeroparque": (-34.5611, -58.4173),
    "Puerto Madero": (-34.6083, -58.3712),
    "Cementerio Recoleta": (-34.5885, -58.3950),
    "Centro de Buenos Aires": (-34.6092, -58.3838),
    "Alto Palermo": (-34.5650, -58.4240),
    "Belgrano": (-34.5750, -58.4400),
    "Caballito": (-34.6090, -58.4350),
    "Palermo Soho": (-34.5760, -58.4300),
    "Ezeiza": (-34.8122, -58.5398),
    "Buquebus": (-34.6037, -58.3816)
}

# Puntos que definen la línea recta imaginaria
linea_punto1 = (-34.582558, -58.381828)
linea_punto2 = (-34.536734, -58.466495)

def recta_corte(lat, lon):
    # Función para determinar si un punto está del lado correcto de la línea
    x, y = lat, lon
    x1, y1 = linea_punto1
    x2, y2 = linea_punto2
    
    # Calculamos el producto cruzado
    cross_product = (y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)
    return cross_product < 0

def agregar_ruido(lat, lon, radio_km=2, ubicacion=None):
    # Convertir km a grados (aprox. 1 grado = 111 km)
    radio_grados = radio_km / 111
    # Para ubicaciones especiales, limitar el offset positivo a 0.5 km
    offset_lat = np.random.uniform(-radio_grados, radio_grados)
    offset_lon = np.random.uniform(-radio_grados, radio_grados)
    if recta_corte(lat+offset_lat, lon+offset_lon):#ubicacion in ["Aeroparque", "Puerto Madero", "Buquebus", "Retiro"]:
        return lat, lon
    else:
        return lat + offset_lat, lon + offset_lon
