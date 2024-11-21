import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Inicio Parte 1

# Cargar el archivo
try:
    df = pd.read_excel("Ventas.xlsx")
except FileNotFoundError:
    st.error("El archivo 'Ventas.xlsx' no se encuentra. Por favor, verifica la ruta.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error inesperado al cargar 'Ventas.xlsx': {e}")
    st.stop()

# Título de Streamlit
st.title("TECStore - Detalle de ventas futuras y prescripción de inventarios")

# Validar columnas necesarias
required_columns = ['Fecha', 'GTIN', 'Piezas', 'Campus', 'Empresa']
if not all(col in df.columns for col in required_columns):
    st.error(f"El archivo no contiene las columnas necesarias: {required_columns}")
    st.stop()

# Filtrar fechas
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
if df['Fecha'].isna().any():
    st.warning("Algunas filas tienen fechas inválidas y serán eliminadas.")
df = df.dropna(subset=['Fecha'])
fecha_inicio = "2022-08-01"
fecha_fin = "2024-08-31"
df = df[(df["Fecha"] >= fecha_inicio) & (df["Fecha"] <= fecha_fin)]

# Filtrar empresa y columnas necesarias
df = df[df['Empresa'] != 'Tecmilenio']
df_TS = df[["Fecha", "GTIN", "Piezas", "Campus"]].dropna()
df_TS = df_TS.set_index('Fecha')

# Agrupar por mes y crear combinaciones de fechas, productos y campus
monthly_df = df_TS.groupby(['GTIN', 'Campus']).resample('M')['Piezas'].sum().reset_index()
full_date_range = pd.date_range(start=monthly_df['Fecha'].min(), end=monthly_df['Fecha'].max(), freq='M')
product_campus_combinations = pd.MultiIndex.from_product(
    [monthly_df['GTIN'].unique(), monthly_df['Campus'].unique(), full_date_range],
    names=['GTIN', 'Campus', 'Fecha']
)
monthly_df = monthly_df.set_index(['GTIN', 'Campus', 'Fecha']).reindex(product_campus_combinations, fill_value=0).reset_index()

# Crear columna de periodo
monthly_df['Periodo'] = monthly_df['Fecha'].dt.to_period('M').astype(str)

# Filtrar los datos para el campus Monterrey
monterrey_df = monthly_df[monthly_df['Campus'] == 'Monterrey']

# Crear una lista para almacenar los pronósticos de este campus
forecast_list = []

# Realizar suavizamiento exponencial para cada grupo (GTIN, Periodo) dentro del campus Monterrey
for (product, period), group in monterrey_df.groupby(['GTIN', 'Periodo']):
    # Asegurarse de que el grupo tenga un índice de fecha
    group = group.set_index('Fecha')['Piezas']
    
    # Verificar datos suficientes para el modelo estacional
    if len(group) < 24:  # Se necesitan al menos 2 años para estacionalidad completa
        st.warning(f"El grupo {product} del campus Monterrey en el periodo {period} tiene menos de 24 datos. Se omite.")
        continue

    # Validar datos: completar faltantes y asegurar frecuencia mensual
    group = group.dropna().asfreq('M', fill_value=0)

    # Inicializar y ajustar el modelo de suavizamiento exponencial
    try:
        model = ExponentialSmoothing(group, trend="add", seasonal="add", seasonal_periods=12)
        fit_model = model.fit()

        # Generar pronóstico para los próximos 3 meses
        forecast = fit_model.forecast(steps=3)

        # Crear un DataFrame para el pronóstico y agregar columnas de identificación
        forecast_df = pd.DataFrame({
            'Periodo': period,
            'GTIN': product,
            'Campus': 'Monterrey',
            'Fecha': forecast.index,
            'Predicted Units Sold': forecast.values
        })

        # Establecer valores negativos en 'Predicted Units Sold' a cero
        forecast_df['Predicted Units Sold'] = forecast_df['Predicted Units Sold'].clip(lower=0)

        # Agregar el DataFrame de pronóstico a la lista
        forecast_list.append(forecast_df)
    except Exception as e:
        st.warning(f"Error al ajustar el modelo para {product}-Monterrey-{period}. Detalle: {e}")
        print(f"Error: {e}")
        continue

# Concatenar todos los DataFrames de pronóstico
if forecast_list:
    consolidated_forecast_df = pd.concat(forecast_list).reset_index(drop=True)
    st.subheader("Pronósticos para el Campus Monterrey")
    st.dataframe(consolidated_forecast_df, use_container_width=True)
else:
    st.warning("No se generaron pronósticos para el campus Monterrey debido a la falta de datos.")

# Final Parte 1
