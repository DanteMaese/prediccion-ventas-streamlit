import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Definir la ruta del archivo como variable para facilidad de ajuste
RUTA_ARCHIVO = "C:\\Users\\dante\\Downloads\\BD Ventas Tec Store - Campus MTY.xlsx"

# --- Funciones de procesamiento con caché ---
@st.cache_data
def cargar_datos():
    """Carga y limpia los datos de Excel, excluyendo registros de 'Tecmilenio'."""
    df = pd.read_excel(RUTA_ARCHIVO)
    print("Columnas disponibles en el archivo:", df.columns)  # Para verificar nombres de columnas
    df = df[df['Empresa'] != 'Tecmilenio']
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
        forecast_df['Predicción de Unidades'] = forecast_df['Predicción de Unidades'].clip(lower=0)
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

# --- Ajustes de formato para la tabla ---
# Asegúrate de que 'GTIN' sea un número entero sin comas
forecast_df['GTIN'] = forecast_df['GTIN'].astype('int64')

# Formatear 'Fecha' para mostrar solo 'YYYY-MM-DD'
forecast_df['Fecha'] = forecast_df['Fecha'].dt.strftime('%Y-%m-%d')

# --- INICIO de Streamlit ---
st.title("Predicción de Ventas - Campus MTY")

# Crear una nueva columna con la combinación de GTIN y Producto
forecast_df['GTIN - Producto'] = forecast_df['GTIN'].astype(str) + " - " + forecast_df['Producto'].astype(str)

# Campo de selección múltiple con instrucciones
productos_seleccionados = st.multiselect(
    "Selecciona uno o varios productos usando la tecla CTRL para buscar productos (GTIN - Producto):",
    options=forecast_df['GTIN - Producto'].unique()
)

# Filtrar el DataFrame para los productos seleccionados
# Extraemos el GTIN de cada producto seleccionado para filtrar
gtins_seleccionados = [int(producto.split(" - ")[0]) for producto in productos_seleccionados]
prediccion_productos = forecast_df[forecast_df['GTIN'].isin(gtins_seleccionados)]

# Mostrar la predicción para los productos seleccionados
if not prediccion_productos.empty:
    st.subheader("Predicción para los Productos Seleccionados")
    st.write(prediccion_productos[['GTIN', 'Producto', 'Categoría', 'Campus', 'Fecha', 'Predicción de Unidades']])
else:
    st.write("No se encontraron predicciones para los productos seleccionados.")

# Mostrar reglas de negocio o notas adicionales
st.write("**Regla de negocio**: La predicción se basa en el análisis de ventas históricas utilizando un modelo de Exponential Smoothing.")
st.write("**Nota**: Las predicciones son aproximadas y pueden variar según estacionalidad y tendencia histórica.")
