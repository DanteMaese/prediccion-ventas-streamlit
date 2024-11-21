import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Inicio Parte 1

# Definir la ruta del archivo como variable para facilidad de ajuste
df = "Ventas.xlsx"

#Titulo de Streamlite
st.title("TECStore - Detalle de ventas futuras y prescripción de inventarios")

# Fechas
fecha_inicio = "2022-08-01"
fecha_fin = "2024-08-31"
df = df[(df["Fecha"] >= fecha_inicio) & (df["Fecha"] <= fecha_fin)]
fecha_minima = df["Fecha"].min()
fecha_maxima = df["Fecha"].max()

# Valores Tecmilenio
df = df[df['Empresa'] != 'Tecmilenio']
df_TS = df[["Fecha", "GTIN", "Piezas", "Campus"]]
df_TS = df_TS.dropna()
df_TS['Fecha'] = pd.to_datetime(df_TS['Fecha'])
df_TS = df_TS.set_index('Fecha')

monthly_df = df_TS.groupby(['GTIN', 'Campus']).resample('M')['Piezas'].sum().reset_index()

# Obtener la fecha mínima y máxima
fecha_minima = monthly_df["Fecha"].min()
fecha_maxima = monthly_df["Fecha"].max()

# Step 2: Define the full date range
#full_date_range = pd.date_range(start=monthly_df['Fecha'].min(), end='2024-08-31', freq='M')
full_date_range = pd.date_range(start=monthly_df['Fecha'].min(), end=monthly_df['Fecha'].max(), freq='M')

# Step 3: Create a MultiIndex for all combinations of Product, Campus, and Date
product_campus_combinations = pd.MultiIndex.from_product(
    [monthly_df['GTIN'].unique(), monthly_df['Campus'].unique(), full_date_range],
    names=['GTIN', 'Campus', 'Fecha']
)

monthly_df = monthly_df.set_index(['GTIN', 'Campus', 'Fecha']).reindex(product_campus_combinations, fill_value=0).reset_index()

monthly_df['Periodo'] = f"{monthly_df['Fecha'].max().year}{monthly_df['Fecha'].max().month:02d}"

monthly_df = monthly_df.set_index('Fecha')

# Define an empty list to store individual forecast DataFrames
forecast_list = []

# Perform exponential smoothing for each (Product, Campus) group
for (product, campus, period), group in monthly_df.groupby(['GTIN', 'Campus','Periodo']):
    # Ensure the group is sorted by date
    group = group.resample('M').sum()  # Resample to ensure monthly frequency
    group = group['Piezas']
    
    # Initialize and fit the exponential smoothing model
    model = ExponentialSmoothing(group, trend="add", seasonal="add", seasonal_periods=12)
    fit_model = model.fit()
    
    # Forecast the next 3 months
    forecast = fit_model.forecast(steps=3)
    
    # Create a DataFrame for the forecast and add identifying columns
    forecast_df = pd.DataFrame({
        'Periodo': period,
        'GTIN': product,
        'Campus': campus,
        'Fecha': forecast.index,
        'Predicted Units Sold': forecast.values
     })

st.subheader("Tabla de Predicciones Consolidadas")
st.dataframe(forecast_df, use_container_width=True)


# Final Parte 1
