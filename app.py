import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Inicio Parte 1

# Definir la ruta del archivo como variable para facilidad de ajuste
RUTA_ARCHIVO = "Ventas.xlsx"

st.title("TECStore - Detalle de ventas futuras y prescripción de inventarios")

@st.cache_data
def cargar_datos():
    """Carga y limpia los datos de Excel, excluyendo registros de 'Tecmilenio'."""
    df = pd.read_excel(RUTA_ARCHIVO)
    print("Columnas disponibles en el archivo:", df.columns)  # Para verificar nombres de columnas
    df = df[df['Empresa'] != 'Tecmilenio']
    df = df[df['Campus'] == 'Monterrey']  # Filtrar solo para Campus Monterrey
    df_TS = df[["Fecha", "GTIN", "Piezas", "Campus"]].dropna()
    df_TS['Fecha'] = pd.to_datetime(df_TS['Fecha'])
    df_TS = df_TS.set_index('Fecha')
    return df, df_TS

@st.cache_data
def procesar_datos(df_TS):
    """Agrupa los datos por mes y crea un DataFrame con todas las combinaciones de fechas, productos y campus."""
    monthly_df = df_TS.groupby(['GTIN', 'Campus']).resample('M')['Piezas'].sum().reset_index()
    full_date_range = pd.date_range(start=monthly_df['Fecha'].min(), end='2024-08-31', freq='M')
    product_campus_combinations = pd.MultiIndex.from_product(
        [monthly_df['GTIN'].unique(), monthly_df['Campus'].unique(), full_date_range],
        names=['GTIN', 'Campus', 'Fecha']
    )
    monthly_df = monthly_df.set_index(['GTIN', 'Campus', 'Fecha']).reindex(product_campus_combinations, fill_value=0).reset_index()
    monthly_df = monthly_df.set_index('Fecha')
    return monthly_df

@st.cache_data
def generar_predicciones(monthly_df):
    """Aplica Exponential Smoothing para predecir las ventas de los próximos 3 meses por cada combinación de producto y campus."""
    forecast_list = []
    for (product, campus), group in monthly_df.groupby(['GTIN', 'Campus']):
        group = group.resample('M').sum()['Piezas']
        model = ExponentialSmoothing(group, trend="add", seasonal="add", seasonal_periods=12)
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=3)
        forecast_df = pd.DataFrame({
            'GTIN': product,
            'Campus': campus,
            'Fecha': forecast.index,
            'Predicción de Unidades': forecast.values
        })
    #     forecast_df['Predicción de Unidades'] = forecast_df['Predicción de Unidades'].clip(lower=0)
    #     forecast_list.append(forecast_df)
    # return pd.concat(forecast_list).reset_index(drop=True)

# Final Parte 1

# Inicio Parte 2

# --- Cargar y procesar los datos usando las funciones cacheadas ---
df, df_TS = cargar_datos()
monthly_df = procesar_datos(df_TS)
forecast_df = generar_predicciones(monthly_df)

# Consolidar las predicciones en columnas por fecha
forecast_consolidado = forecast_df.pivot_table(
    index=['GTIN', 'Campus'],             # Indexamos por GTIN y Campus
    columns='Fecha',                      # Las fechas se convierten en columnas
    values='Predicción de Unidades',      # Usar las predicciones como valores
    aggfunc='sum'                         # Sumar valores si hay duplicados
).reset_index()

# Renombrar columnas según las fechas mapeadas
fechas_mapeo = {
    pd.Timestamp('2024-09-30'): 'Pred. Sep 2024',
    pd.Timestamp('2024-10-31'): 'Pred. Oct 2024',
    pd.Timestamp('2024-11-30'): 'Pred. Nov 2024'
}
forecast_consolidado.rename(columns=fechas_mapeo, inplace=True)

st.subheader("Tabla de Predicciones Consolidadas")
st.dataframe(forecast_consolidado, use_container_width=True)

# Final Parte 2
