import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Definir la ruta del archivo como variable para facilidad de ajuste
RUTA_ARCHIVO = "BD Ventas Tec Store - Campus MTY.xlsx"

# --- Funciones de procesamiento con caché ---
@st.cache_data
def cargar_datos():
    # Cargar y limpiar los datos
    df = pd.read_excel("C:\\Users\\dante\\Downloads\\BD Ventas Tec Store - Campus MTY.xlsx")
    df = df[df['Empresa'] != 'Tecmilenio']
    df_TS = df[["Fecha", "GTIN", "Piezas", "Campus"]].dropna()
    df_TS['Fecha'] = pd.to_datetime(df_TS['Fecha'])
    df_TS = df_TS.set_index('Fecha')
    return df_TS

@st.cache_data
def procesar_datos(df_TS):
    # Crear el DataFrame mensual y realizar el preprocesamiento
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
    # Crear lista para almacenar predicciones y aplicar el modelo de predicción
    forecast_list = []
    for (product, campus), group in monthly_df.groupby(['GTIN', 'Campus']):
        group = group.resample('M').sum()['Piezas']
        model = ExponentialSmoothing(group, trend="add", seasonal="add", seasonal_periods=12)
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=3)
        forecast_df = pd.DataFrame({
            'Product': product,
            'Campus': campus,
            'Date': forecast.index,
            'Predicted Units Sold': forecast.values
        })
        forecast_df['Predicted Units Sold'] = forecast_df['Predicted Units Sold'].clip(lower=0)
        forecast_list.append(forecast_df)
    return pd.concat(forecast_list).reset_index(drop=True)

# --- Cargar y procesar los datos usando las funciones cacheadas ---
df, df_TS = cargar_datos()
monthly_df = procesar_datos(df_TS)
forecast_df = generar_predicciones(monthly_df)

# Extraer las columnas únicas de GTIN, Producto y Categoría para el join
info_producto = df[['GTIN', 'Producto', 'Categoría']].drop_duplicates()

# Realizar el join para agregar Producto y Categoría a forecast_df
forecast_df = forecast_df.merge(info_producto, on='GTIN', how='left')

# --- Cargar el archivo de stock y realizar el join ---
@st.cache_data
def cargar_stock():
    """Carga los datos de stock desde el archivo BD Stock.xlsx."""
    stock_df = pd.read_excel("BD Stock.xlsx")
    stock_df['GTIN'] = stock_df['GTIN'].astype('int64')  # Convertir GTIN a int
    return stock_df

# Cargar los datos de stock
stock_df = cargar_stock()

# Asegurarse de que GTIN en forecast_df sea del mismo tipo
forecast_df['GTIN'] = forecast_df['GTIN'].astype('int64')

# Realizar el join para agregar la columna de stock
forecast_df = forecast_df.merge(stock_df[['GTIN', 'Stock']], on='GTIN', how='left')

# --- INICIO de Streamlit ---
st.title("Predicción de Ventas - Campus MTY")

# Filtro 1: Selección de productos
st.subheader("Filtrar por Producto")
productos_seleccionados = st.multiselect(
    "Escribe el nombre de un producto, selecciona uno o varios productos de la lista.",
    options=forecast_df['Producto'].unique()
)

# Filtro 2: Selección de categorías
st.subheader("Filtrar por Categoría")
categorias_seleccionadas = st.multiselect(
    "Selecciona una o varias categorías de la lista.",
    options=forecast_df['Categoría'].unique()
)

# Filtrar el DataFrame según los productos y las categorías seleccionadas
prediccion_productos = forecast_df.copy()

if productos_seleccionados:
    prediccion_productos = prediccion_productos[prediccion_productos['Producto'].isin(productos_seleccionados)]

if categorias_seleccionadas:
    prediccion_productos = prediccion_productos[prediccion_productos['Categoría'].isin(categorias_seleccionadas)]

# Mostrar la predicción para los productos seleccionados con formato mejorado
if not prediccion_productos.empty:
    st.subheader("Predicción para los Productos Seleccionados")
    columnas_para_mostrar = ['GTIN', 'Producto', 'Categoría', 'Campus', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024', 'Stock']
    styled_df = prediccion_productos[columnas_para_mostrar].style.format(
        {
            'Septiembre 2024': '{:.0f}',
            'Octubre 2024': '{:.0f}',
            'Noviembre 2024': '{:.0f}',
            'Stock': '{:.0f}'
        }
    ).hide(axis="index")  # Ocultar el índice
    st.dataframe(styled_df, use_container_width=True)
else:
    st.write("No se encontraron predicciones para los filtros seleccionados.")
