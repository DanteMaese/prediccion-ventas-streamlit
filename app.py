import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Rutas de archivo
RUTA_ARCHIVO = "BD Ventas Tec Store - Campus MTY.xlsx"
RUTA_STOCK = "BD Stock.xlsx"

# Verificar existencia de archivos
if not os.path.exists(RUTA_ARCHIVO):
    st.error(f"Archivo no encontrado: {RUTA_ARCHIVO}")
if not os.path.exists(RUTA_STOCK):
    st.error(f"Archivo no encontrado: {RUTA_STOCK}")

# Funciones cacheadas
@st.cache_data
def cargar_datos():
    df = pd.read_excel(RUTA_ARCHIVO)
    df = df[df['Empresa'] != 'Tecmilenio']
    df_TS = df[["Fecha", "GTIN", "Piezas", "Campus"]].dropna()
    df_TS['Fecha'] = pd.to_datetime(df_TS['Fecha'])
    return df, df_TS

@st.cache_data
def cargar_stock():
    stock_df = pd.read_excel(RUTA_STOCK)
    stock_df['GTIN'] = stock_df['GTIN'].astype('int64')
    return stock_df

@st.cache_data
def procesar_datos(df_TS):
    monthly_df = df_TS.groupby(['GTIN', 'Campus']).resample('M')['Piezas'].sum().reset_index()
    return monthly_df

@st.cache_data
def generar_predicciones(monthly_df):
    forecast_list = []
    for (product, campus), group in monthly_df.groupby(['GTIN', 'Campus']):
        group = group.set_index('Fecha').resample('M').sum()['Piezas']
        model = ExponentialSmoothing(group, trend="add", seasonal="add", seasonal_periods=12)
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=3)
        forecast_df = pd.DataFrame({
            'GTIN': product,
            'Campus': campus,
            'Fecha': forecast.index,
            'Predicción de Unidades': forecast.values
        })
        forecast_list.append(forecast_df)
    return pd.concat(forecast_list).reset_index(drop=True)

# Procesar datos
df, df_TS = cargar_datos()
monthly_df = procesar_datos(df_TS)
forecast_df = generar_predicciones(monthly_df)

# Join con información adicional
info_producto = df[['GTIN', 'Producto', 'Categoría']].drop_duplicates()
forecast_df = forecast_df.merge(info_producto, on='GTIN', how='left')

# Pivot y join con stock
forecast_pivot = forecast_df.pivot(index=['GTIN', 'Producto', 'Categoría', 'Campus'], columns='Fecha', values='Predicción de Unidades').reset_index()
forecast_pivot.rename(columns={
    '2024-09-30': 'Septiembre 2024',
    '2024-10-31': 'Octubre 2024',
    '2024-11-30': 'Noviembre 2024'
}, inplace=True)
stock_df = cargar_stock()
forecast_pivot = forecast_pivot.merge(stock_df[['GTIN', 'Stock']], on='GTIN', how='left')

# Streamlit
st.title("Predicción de Ventas - Campus MTY")
productos_seleccionados = st.multiselect("Selecciona productos:", forecast_pivot['Producto'].unique())
categorias_seleccionadas = st.multiselect("Selecciona categorías:", forecast_pivot['Categoría'].unique())
prediccion_productos = forecast_pivot.copy()

if productos_seleccionados:
    prediccion_productos = prediccion_productos[prediccion_productos['Producto'].isin(productos_seleccionados)]
if categorias_seleccionadas:
    prediccion_productos = prediccion_productos[prediccion_productos['Categoría'].isin(categorias_seleccionadas)]

if not prediccion_productos.empty:
    st.dataframe(prediccion_productos, use_container_width=True)
else:
    st.write("No se encontraron resultados.")
