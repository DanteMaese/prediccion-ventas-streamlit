import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Inicio Parte 1

# Cargar el archivo
df = pd.read_excel("Ventas.xlsx")

# Título de Streamlit
st.title("TECStore - Detalle de ventas futuras y prescripción de inventarios")

# Filtrar fechas
fecha_inicio = "2022-08-01"
fecha_fin = "2024-08-31"
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
df = df.dropna(subset=['Fecha'])  # Eliminar fechas inválidas
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

# Definir lista para almacenar predicciones
forecast_list = []

# Realizar predicciones con Exponential Smoothing
for (product, campus, period), group in monthly_df.groupby(['GTIN', 'Campus', 'Periodo']):
    group = group.set_index('Fecha')['Piezas']
    
    # Modelo de predicción
    model = ExponentialSmoothing(group, trend="add", seasonal="add", seasonal_periods=12)
    fit_model = model.fit()
    forecast = fit_model.forecast(steps=3)
    
    # Crear DataFrame para predicciones
    forecast_df = pd.DataFrame({
        'Periodo': period,
        'GTIN': product,
        'Campus': campus,
        'Fecha': forecast.index,
        'Predicted Units Sold': forecast.values
    })
    forecast_list.append(forecast_df)

# Consolidar predicciones
consolidated_forecast_df = pd.concat(forecast_list, ignore_index=True)

# Mostrar resultados en Streamlit
st.subheader("Tabla de Predicciones Consolidadas")
st.dataframe(consolidated_forecast_df, use_container_width=True)

# Final Parte 1
