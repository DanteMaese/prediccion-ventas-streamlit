import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Definir la ruta del archivo como variable para facilidad de ajuste
RUTA_ARCHIVO = "BD Ventas Tec Store - Campus MTY.xlsx"

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

import pandas as pd

# --- Cargar el archivo de stock y realizar el join ---
@st.cache_data
def cargar_stock():
    """Carga los datos de stock desde el archivo BD Stock.xlsx y ajusta el formato de GTIN."""
    stock_df = pd.read_excel("BD Stock.xlsx")
    stock_df['GTIN'] = stock_df['GTIN'].astype('int64')  # Convertir GTIN a int
    return stock_df

# Cargar los datos de stock
stock_df = cargar_stock()

# Asegurarse de que GTIN en forecast_df sea del mismo tipo
forecast_df['GTIN'] = forecast_df['GTIN'].astype('int64')  # Convertir GTIN a int

# Debugging para verificar las claves
print("Valores únicos en stock_df['GTIN']:", stock_df['GTIN'].unique())
print("Valores únicos en forecast_df['GTIN']:", forecast_df['GTIN'].unique())

# Realizar el join para agregar la columna de stock a forecast_df
forecast_df = forecast_df.merge(stock_df[['GTIN', 'Stock']], on='GTIN', how='left')

# Verificar el resultado del join
print("Preview de forecast_df después del join:")
print(forecast_df[['GTIN', 'Stock']].head())

# --- INICIO de Streamlit ---
st.title("Predicción de Ventas - Campus MTY")

# Crear una nueva columna con la combinación de GTIN y Producto
forecast_df['GTIN - Producto'] = forecast_df['GTIN'].astype(str) + " - " + forecast_df['Producto'].astype(str)

# Campo de selección múltiple con instrucciones
productos_seleccionados = st.multiselect(
    "Escribe el nombre de un producto, selecciona uno o varios productos de la lista.",
    options=forecast_df['GTIN - Producto'].unique()
)

# Filtrar el DataFrame para los productos seleccionados
# Extraemos el GTIN de cada producto seleccionado para filtrar
gtins_seleccionados = [int(producto.split(" - ")[0]) for producto in productos_seleccionados]
prediccion_productos = forecast_df[forecast_df['GTIN'].isin(gtins_seleccionados)]

# Mostrar la predicción para los productos seleccionados con formato mejorado
if not prediccion_productos.empty:
    st.subheader("Predicción para los Productos Seleccionados")
    
    # Seleccionar columnas relevantes
    columnas_para_mostrar = ['GTIN', 'Producto', 'Categoría', 'Campus', 'Fecha', 'Predicción de Unidades', 'Stock']
    
    # Crear un DataFrame estilizado para eliminar el índice y ajustar el formato
    styled_df = prediccion_productos[columnas_para_mostrar].style.format(
        {
            'Predicción de Unidades': '{:.0f}',  # Formatear sin decimales
            'Stock': '{:.0f}'  # Formatear sin decimales
        }
    ).hide(axis="index")  # Ocultar el índice

    # Mostrar la tabla en Streamlit usando todo el ancho del contenedor
    st.dataframe(styled_df, use_container_width=True)
else:
    st.write("No se encontraron predicciones para los productos seleccionados.")

# --- Gráfico 1: Comparación de Stock vs Predicciones por Categoría ---
st.subheader("Comparación de Stock vs Predicciones por Categoría")

# Agrupar datos por Categoría
comparacion_categoria = prediccion_productos.groupby('Categoría')[['Stock', 'Predicción de Unidades']].sum().reset_index()

# Mostrar gráfico de barras agrupadas
st.bar_chart(data=comparacion_categoria.set_index('Categoría'))

# --- Gráfico 2: Categorías con Riesgo de Quedarse Sin Stock ---
st.subheader("Categorías con Riesgo de Quedarse Sin Stock")

# Identificar categorías en riesgo
categorias_riesgo = comparacion_categoria[comparacion_categoria['Predicción de Unidades'] > comparacion_categoria['Stock']]

if not categorias_riesgo.empty:
    st.bar_chart(data=categorias_riesgo.set_index('Categoría')[['Predicción de Unidades']])
else:
    st.write("No hay categorías con riesgo de quedarse sin stock.")
