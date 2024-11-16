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

# --- Reestructuración de forecast_df ---
forecast_pivot = forecast_df.pivot_table(
    index=['GTIN', 'Producto', 'Categoría', 'Campus'],
    columns='Fecha',
    values='Predicción de Unidades',
    aggfunc='sum'
).fillna(0).reset_index()

# Renombrar columnas si las fechas existen
fechas_mapeo = {
    '2024-09-30': 'Septiembre 2024',
    '2024-10-31': 'Octubre 2024',
    '2024-11-30': 'Noviembre 2024'
}

forecast_pivot.rename(columns={col: fechas_mapeo[col] for col in fechas_mapeo if col in forecast_pivot.columns}, inplace=True)

# Validar que las columnas de predicción existan
for columna in ['Septiembre 2024', 'Octubre 2024', 'Noviembre 2024']:
    if columna not in forecast_pivot.columns:
        forecast_pivot[columna] = 0

# --- INICIO de Streamlit ---
st.title("Predicción de Ventas - Campus MTY")

# Filtro 1: Selección de productos
st.subheader("Filtrar por Producto")
productos_seleccionados = st.multiselect(
    "Escribe el nombre de un producto, selecciona uno o varios productos de la lista.",
    options=forecast_pivot['Producto'].unique()
)

# Filtro 2: Selección de categorías
st.subheader("Filtrar por Categoría")
categorias_seleccionadas = st.multiselect(
    "Selecciona una o varias categorías de la lista.",
    options=forecast_pivot['Categoría'].unique()
)

# Filtrar el DataFrame según los productos y las categorías seleccionadas
prediccion_productos = forecast_pivot.copy()

if productos_seleccionados:
    prediccion_productos = prediccion_productos[prediccion_productos['Producto'].isin(productos_seleccionados)]

if categorias_seleccionadas:
    prediccion_productos = prediccion_productos[prediccion_productos['Categoría'].isin(categorias_seleccionadas)]

# Mostrar la predicción para los productos seleccionados con formato mejorado
if not prediccion_productos.empty:
    st.subheader("Predicción para los Productos Seleccionados")
    
    # Seleccionar columnas relevantes
    columnas_para_mostrar = ['GTIN', 'Producto', 'Categoría', 'Campus', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024', 'Stock']
    columnas_existentes = [col for col in columnas_para_mostrar if col in prediccion_productos.columns]
    
    styled_df = prediccion_productos[columnas_existentes].style.format(
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

# --- Gráfico: Comparación de Stock vs Predicciones por Producto ---
st.subheader("Comparación de Stock vs Predicciones por Producto")

# Agrupar datos por Producto
comparacion_producto = prediccion_productos.groupby('Producto')[['Stock', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024']].sum().reset_index()

# Mostrar gráfico de barras agrupadas
st.bar_chart(data=comparacion_producto.set_index('Producto'))

# --- Gráfico 2: Categorías con Riesgo de Quedarse Sin Stock ---
st.subheader("Categorías con Riesgo de Quedarse Sin Stock")

# Identificar categorías en riesgo
comparacion_categoria = prediccion_productos.groupby('Categoría')[['Stock', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024']].sum().reset_index()
comparacion_categoria['Predicción Total'] = comparacion_categoria[['Septiembre 2024', 'Octubre 2024', 'Noviembre 2024']].sum(axis=1)
categorias_riesgo = comparacion_categoria[comparacion_categoria['Predicción Total'] > comparacion_categoria['Stock']]

if not categorias_riesgo.empty:
    st.bar_chart(data=categorias_riesgo.set_index('Categoría')[['Predicción Total']])
else:
    st.write("No hay categorías con riesgo de quedarse sin stock.")
