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
# Crear columnas para cada mes de predicción
fechas_mapeo = {
    '2024-09-30': 'Pred. Sep 2024',
    '2024-10-31': 'Pred. Oct 2024',
    '2024-11-30': 'Pred. Nov 2024'
}

# Inicializar las columnas de predicción en el DataFrame
for col in fechas_mapeo.values():
    forecast_df[col] = 0

# Llenar las columnas de predicción con los valores correspondientes
for fecha, col_name in fechas_mapeo.items():
    forecast_df.loc[forecast_df['Fecha'] == fecha, col_name] = forecast_df['Predicción de Unidades']

# Consolidar las filas para agrupar por producto, campus y categoría
forecast_consolidado = forecast_df.groupby(['GTIN', 'Producto', 'Categoría', 'Campus'], as_index=False).agg(
    {
        'Septiembre 2024': 'sum',
        'Octubre 2024': 'sum',
        'Noviembre 2024': 'sum'
    }
)

# --- Cargar el archivo de stock y realizar el join ---
@st.cache_data
def cargar_stock():
    """Carga los datos de stock desde el archivo BD Stock.xlsx."""
    stock_df = pd.read_excel("BD Stock.xlsx")
    stock_df['GTIN'] = stock_df['GTIN'].astype('int64')  # Convertir GTIN a int
    return stock_df

# Cargar los datos de stock
stock_df = cargar_stock()

# Asegurarse de que GTIN en forecast_consolidado sea del mismo tipo
forecast_consolidado['GTIN'] = forecast_consolidado['GTIN'].astype('int64')

# Realizar el join para agregar la columna de stock
forecast_consolidado = forecast_consolidado.merge(stock_df[['GTIN', 'Stock']], on='GTIN', how='left')

# --- INICIO de Streamlit ---
st.title("Predicción de Ventas - Campus MTY")

# Campo de selección múltiple con instrucciones
productos_seleccionados = st.multiselect(
    "Escribe el nombre de un producto, selecciona uno o varios productos de la lista.",
    options=forecast_consolidado['Producto'].unique()
)

# Filtrar el DataFrame para los productos seleccionados
prediccion_productos = forecast_consolidado[forecast_consolidado['Producto'].isin(productos_seleccionados)]

# Mostrar la predicción para los productos seleccionados con formato mejorado
if not prediccion_productos.empty:
    st.subheader("Predicción para los Productos Seleccionados")
    
    # Seleccionar columnas relevantes
    columnas_para_mostrar = ['GTIN', 'Producto', 'Categoría', 'Campus', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024', 'Stock']
    
    styled_df = prediccion_productos[columnas_para_mostrar].style.format(
        {
            'Pred. Sep 2024': '{:.0f}',
            'Pred. Oct 2024': '{:.0f}',
            'Pred. Nov 2024': '{:.0f}',
            'Stock': '{:.0f}'
        }
    ).hide(axis="index")  # Ocultar el índice

    st.dataframe(styled_df, use_container_width=True)
else:
    st.write("No se encontraron predicciones para los productos seleccionados.")

# --- Gráfico 1: Comparación de Stock vs Predicciones por Categoría ---
st.subheader("Comparación de Stock vs Predicciones por Categoría")
comparacion_categoria = forecast_pivot.groupby('Categoría')[['Stock', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024']].sum().reset_index()
st.bar_chart(data=comparacion_categoria.set_index('Categoría'))
