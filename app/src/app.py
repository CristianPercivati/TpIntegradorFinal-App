import streamlit as st
import requests
import random
import pandas as pd
from io import StringIO
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from utils import ubicaciones_coords, agregar_ruido
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="TP Integrador - BATransf", page_icon="🚖", layout="wide")
st.title("Sistema de Optimización de Tarifas - BATransf")

menu = ["Train", "Show"]

st.sidebar.title("Menú")
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Train":
    res = requests.get("http://localhost:8000/get-data")
    data = st.selectbox("Seleccione el dataset", res.json()["data"])
    st.subheader("Entrenar el modelo")
    model_name = st.text_input("Ingrese el nombre del modelo","Modelo 1")
    col1,col2,col3,col4,col5=st.columns(5)
    episodes = col1.number_input("Ingrese el número de episodios", min_value=1, step=1, value=1)
    days = col2.number_input("Ingrese el número de días", min_value=1, max_value=365, step=1, value=5)
    if st.button("Comenzar el entrenamiento"):
        with st.spinner("Entrenando..."):
            
            response = requests.post("http://localhost:8000/train", 
                                     json={
                                         "model_name": model_name, 
                                         "episodes": episodes, 
                                         "days": days},
                                         stream=True)
            
            progress_bar = st.progress(0)
            for chunk in response.iter_lines():
                if chunk:
                    data = eval(chunk.decode("utf-8"))
                    progress = data["progress"]
                    progress_bar.progress(int(progress))
                    if progress == 100:
                        st.write("Entrenamiento completo!")

elif choice == "Show":
    st.subheader("Mostrar resultados")
    #Leer los modelos de la carpeta models:
    res = requests.get("http://localhost:8000/models")
    model = st.selectbox("Seleccione el modelo", res.json()["models"])
    if st.button("Cargar"):
        
        response = requests.get("http://localhost:8000/output?model_name="+model, stream=True)
        if response.status_code == 200:
            # Convertir el contenido CSV en un DataFrame
            csv_content = response.text
            df_2025 = pd.read_csv(StringIO(csv_content))
            df = pd.read_csv('data\\df_final.csv')

            tabs = st.tabs(["Head del CSV", "Pronóstico de Ingresos", "Mapa Interactivo"])

            with tabs[0]:
                st.write("Head del CSV:")
                st.dataframe(df_2025.head())
                st.write("Tail del CSV:")
                st.dataframe(df_2025.tail())

            with tabs[1]:
                df_series = df.copy()
                df_series['fecha'] = pd.to_datetime(df_series['Fecha'], format='%d/%m/%Y')
                df_series = df_series.groupby('fecha')['total_usd'].sum().reset_index()
                df_series = df_series.set_index('fecha')
                df_series = df_series.asfreq('D').fillna(method='ffill')

                #Reemplazamos los ceros por un pequeño valor (por ejemplo, el 1% de la media)
                df_series['total_usd'] = df_series['total_usd'].replace(0, df_series['total_usd'].mean() * 0.01)

                #Modelo que utilizamos para el forecasting
                ets_model = ExponentialSmoothing(df_series['total_usd'], trend='add', seasonal='add', seasonal_periods=260)
                ets_result = ets_model.fit()
                ets_forecast = ets_result.forecast(steps=365)
                ets_forecast_index = pd.date_range(start=df_series.index[-1] + pd.Timedelta(days=1), periods=365, freq='D')
                ets_forecast_df = pd.DataFrame(ets_forecast, index=ets_forecast_index, columns=['forecast'])

                df_2025_series = df_2025.copy()
                df_2025_series['fecha'] = pd.to_datetime(df_2025_series['fecha'], format='%d/%m/%Y')
                df_2025_series = df_2025_series.groupby('fecha')['Total'].sum().reset_index()
                df_2025_series = df_2025_series.set_index('fecha')
                df_2025_series = df_2025_series.asfreq('D').fillna(method='ffill')
                df_2025_series['Total'] = df_2025_series['Total'].replace(0, df_2025_series['Total'].mean() * 0.01)

                media_historica = df_series['total_usd'].mean()
                media_pronostico = df_2025_series['Total'].mean()
                # Aplicar suavizado con ventana de 7 días:
                df_series['total_usd_smooth'] = df_series['total_usd'].rolling(window=7, min_periods=1).mean()
                ets_forecast_df['forecast_smooth'] = ets_forecast_df['forecast'].rolling(window=7, min_periods=1).mean()
                df_2025_series['Total_smooth'] = df_2025_series['Total'].rolling(window=7, min_periods=1).mean()

                # Crear la figura del gráfico:
                plt.figure(figsize=(10, 6))
                plt.plot(df_series.index, df_series['total_usd_smooth'], label='Serie Histórica', color='blue')
                plt.plot(ets_forecast_df.index, ets_forecast_df['forecast_smooth'], label='Pronóstico', color='red', linestyle='--')
                plt.plot(df_2025_series.index, df_2025_series['Total_smooth'], label='Pronóstico 2025', color='green')
                plt.title('Pronóstico de Ingresos para 2025')
                plt.xlabel('Fecha')
                plt.ylabel('Ingresos Total USD')
                plt.legend()

                # Mostrar el gráfico en Streamlit
                st.pyplot(plt.gcf())
                st.write(f"Media histórica: {media_historica:.2f}")
                st.write(f"Media pronóstico 2025: {media_pronostico:.2f}")
            
            with tabs[2]:
                df_2025['lat'] = df_2025['Ubicación'].apply(lambda x: agregar_ruido(*ubicaciones_coords[x])[0] if x in ubicaciones_coords else None)
                df_2025['lon'] = df_2025['Ubicación'].apply(lambda x: agregar_ruido(*ubicaciones_coords[x])[1] if x in ubicaciones_coords else None)
            
                # Convertir la columna 'Fecha' a datetime
                df_2025['Fecha'] = pd.to_datetime(df_2025['Fecha'], format='%d/%m/%Y')
                # Crear el mapa
                fig = px.scatter_mapbox(
                    df_2025,
                    lat="lat",
                    lon="lon",
                    color="Aceptado",
                    color_discrete_map={True: "orange", False: "blue"},  # Naranja para concretados, azul para posibles
                    hover_name="Pasajero",
                    hover_data=["Costo", "Tipo_Servicio", "Ubicación"],
                    animation_frame="Fecha",  # Slider para la evolución día a día
                    zoom=10,
                    height=600,
                    title="Viajes en Buenos Aires (2025)",
                )
                # Configurar el estilo del mapa
                fig.update_layout(
                    mapbox_style="carto-darkmatter",
                    mapbox_center={"lat": -34.6037, "lon": -58.3816},  # Centrar en Buenos Aires
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                )
                # Si no hay viajes tomados, por plotly tengo que inventar algún punto true
                if not df_2025[df_2025["Fecha"] == df_2025["Fecha"].iloc[0]]["Aceptado"].any():
                    fig.add_trace(go.Scattermapbox(
                        lat=[None], lon=[None],  # No queremos puntos reales, solo leyenda
                        marker=dict(color="orange"),
                        name="True",
                        showlegend=True
                    ))
                    
                # Mostrar el mapa
                st.plotly_chart(fig)
        else:
            st.error("No se encontró el archivo del modelo seleccionado.")