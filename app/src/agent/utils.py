import pandas as pd 
import ast
from datetime import datetime, timedelta
import numpy as np
import random

df_clima = pd.read_csv('serving\\app\\src\\data\\eda\\clima_dist.csv')
df_servicio = pd.read_csv('serving\\app\\src\\data\\eda\\tipo_servicio.csv')
df_estacionalidad = pd.read_csv('serving\\app\\src\\data\\eda\\estacionalidad.csv')
df_trafico = pd.read_csv('serving\\app\\src\\data\\eda\\trafico_dist.csv')
df_zonas = pd.read_csv('serving\\app\\src\\data\\eda\\zonas.csv')

#Necesitamos convertir estos df a diccionarios por simplicidad para entrenar el modelo

distribucion_clima = dict(zip(df_clima['clima'], df_clima['prob']))
distribucion_clima = {k: v for k, v in distribucion_clima.items() if pd.notna(k)}
seasonality_normalized = df_estacionalidad.set_index('fecha')['seasonal']
distribucion_trafico = dict(zip(df_trafico['trafico'], df_trafico['prob']))
distribucion_trafico = {k: v for k, v in distribucion_trafico.items() if pd.notna(k)}

probabilidad = df_zonas['count'] / df_zonas['count'].sum()

distribucion_zonas = {
    zona: [
        probabilidad.iloc[i],  # Probabilidad
        row['zona_mean'],      # Tarifa base
        row['costo'],          # Costo base
        row['prob_peaje'],     # Probabilidad de peaje
        row['activa'],         # Activa
        row['nombre']          # Nombre
    ]
    for i, (zona, row) in enumerate(df_zonas.iterrows())
}
distribucion_zonas = {k+1: v for k, v in distribucion_zonas.items() if pd.notna(k)}
df_servicio['prob_y_costo_tipo_servicio'] = df_servicio['prob_y_costo_tipo_servicio'].apply(ast.literal_eval)
flota_posible = {
    row['tipo_servicio']: [
        row['prob_y_costo_tipo_servicio'][0], row['prob_y_costo_tipo_servicio'][1]
        ]
        for _, (vehiculo,row) in enumerate(df_servicio.iterrows())
                }

df = pd.read_csv('serving\\app\\src\\data\\df_final.csv')

df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
df = df.dropna(subset=['fecha'])

historico = {}
for _, row in df.iterrows():
    fecha_key = (row['fecha'] - pd.Timedelta(days=row['fecha'].weekday()))
    historico[fecha_key] = {
        'clima': row['clima'],
        'trafico': row['trafico'],
        'feriado': row['feriado']
    }

def generar_datos_2025(pasajero, dia_simulado, fecha, tarifa, vehiculo, aceptado):
            zonas = {
            1: 'Aeroparque',
            2: 'Centro de Buenos Aires',
            3: 'Recoleta',
            4: 'Palermo Soho',
            5: 'Almagro',
            6: 'Alto Palermo',
            7: 'Retiro',
            8: 'Puerto Madero',
            9: 'Belgrano',
            10: 'La Boca',
            11: 'Caballito',
            12: 'Recoleta',
            13: 'Cementerio Recoleta',
            14: 'Almagro',
            15: 'Hipódromo Palermo',
            16: 'San Cristóbal',
            17: 'Desconocido',
            18: 'Desconocido',
            19: 'Desconocido'
            }

            fecha_inicio = datetime(2025, 1, 1).date()
            fecha = fecha.replace(year=fecha_inicio.year)
            nueva_fecha_inicio = datetime(2024, 11, 18).date()

            #Calcular la diferencia de días y ajustar la fecha
            diferencia_dias = (fecha_inicio - nueva_fecha_inicio).days
            fecha = fecha - timedelta(days=diferencia_dias)
            #Desplazar la fecha de inicio a 2024-11-18. Una pequeña "trampa"
            #que nos permite que la serie se visualice de forma continua dado
            #a que el df original no está completo en sus fechas.

            fila = {
                        "Fecha": fecha.strftime("%d/%m/%Y"),
                        "Pasajero": '',
                        "Costo": tarifa,
                        "Peajes": distribucion_zonas[pasajero.zona][3] if random.random() < 0.3 else np.nan,
                        "Espera": np.random.randint(0, 30) if aceptado else np.nan,
                        "Parking": np.nan,  # No simulado
                        "Extra": np.nan,    # No simulado
                        "Total": tarifa if aceptado else 0,
                        "cantidad_pasajeros": random.randint(1, 4),
                        "Tipo_Servicio": vehiculo,
                        "Ubicación": zonas[pasajero.zona],
                        "venta": np.random.uniform(1300, 1500),
                        "fecha": fecha.strftime("%d/%m/%Y"),
                        "total_usd": tarifa,
                        "clima": dia_simulado.clima,
                        "temperatura": np.random.uniform(10, 35),
                        "autopista": "AU 9 de Julio Sur",
                        "trafico": ["Bajo", "Medio", "Alto"][dia_simulado.trafico_level],
                        "feriado": "Sí" if dia_simulado.es_feriado else "No",
                        "genero": np.random.choice(["F", "M", "U"]),
                        "Rating": 0,
                        "Review": '',
                        "Aceptado": True if aceptado else False
                    }
            return fila

def generar_pasajeros(fecha, base_passengers=30, std_dev=5):
            # Convertir la fecha a datetime
            if isinstance(fecha, str):
                fecha = datetime.strptime(fecha, "%d-%m-%Y")

            # Obtener el componente de estacionalidad para la fecha (si existe)
            if fecha in seasonality_normalized.index:
                seasonal_value = seasonality_normalized[fecha]
            else:
                # Si la fecha no está en los datos, usar el mes para aproximar
                mes = fecha.month
                seasonal_value = seasonality_normalized[seasonality_normalized.index.month == mes].mean()

            adjusted_passengers = base_passengers * (1 + seasonal_value)

            # Generar número aleatorio usando la distribución normal
            random_passengers = np.random.normal(adjusted_passengers, std_dev)

            return max(0, int(np.round(random_passengers)))

