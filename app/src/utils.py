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

def agregar_ruido(lat, lon, radio_km=2, ubicacion=None):
    # Convertir km a grados (aprox. 1 grado = 111 km)
    radio_grados = radio_km / 111
    # Para ubicaciones especiales, limitar el offset positivo a 0.5 km
    if ubicacion in ["Aeroparque", "Puerto Madero", "Buquebus", "Retiro"]:
        return lat, lon
        pos_max = 0.5 / 111  # 0.5 km en grados
        offset_lat = np.random.uniform(-radio_grados, radio_grados)
        if offset_lat > pos_max:
            offset_lat = pos_max
        offset_lon = np.random.uniform(-radio_grados, radio_grados)
        if offset_lon > pos_max:
            offset_lon = pos_max-25
    else:
        offset_lat = np.random.uniform(-radio_grados, radio_grados)
        offset_lon = np.random.uniform(-radio_grados, radio_grados)
    return lat + offset_lat, lon + offset_lon
