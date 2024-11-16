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

#######

# Validar que las columnas necesarias existen en forecast_df
columnas_requeridas = ['Fecha', 'Prediccion de Unidades']
for col in columnas_requeridas:
    if col not in forecast_df.columns:
        print("Columnas disponibles:", forecast_df.columns)
        raise ValueError(f"La columna {col} no existe en forecast_df. Revisa los nombres de las columnas.")

# Reestructuración de forecast_df
fechas_mapeo = {
    '2024-09-30': 'Septiembre 2024',
    '2024-10-31': 'Octubre 2024',
    '2024-11-30': 'Noviembre 2024'
}

forecast_consolidado = forecast_df.copy()

# Convertir las fechas al formato legible
forecast_consolidado['Fecha'] = forecast_consolidado['Fecha'].dt.strftime('%Y-%m-%d')
forecast_consolidado['Fecha'] = forecast_consolidado['Fecha'].map(fechas_mapeo)

# Agrupar y convertir filas en columnas
forecast_consolidado = forecast_consolidado.groupby(['GTIN', 'Campus'], as_index=False).agg(
    Producto=('GTIN', 'first'),
    Categoria=('Campus', 'first'),
    **{col: ('Prediccion de Unidades', lambda x: x.iloc[i]) for i, col in enumerate(fechas_mapeo.values())}
)

# Asegurarse de que GTIN sea del mismo tipo en ambos DataFrames
forecast_consolidado['GTIN'] = forecast_consolidado['GTIN'].astype('int64')
stock_df['GTIN'] = stock_df['GTIN'].astype('int64')

# Agregar la columna de stock
forecast_consolidado = forecast_consolidado.merge(stock_df[['GTIN', 'Stock']], on='GTIN', how='left')

# --- INICIO de Streamlit ---
st.title("Predicción de Ventas - Campus MTY")

# Filtrar por productos
productos_seleccionados = st.multiselect(
    "Escribe el nombre de un producto, selecciona uno o varios productos de la lista.",
    options=forecast_consolidado['Producto'].unique()
)

# Filtrar por categorías
categorias_seleccionadas = st.multiselect(
    "Selecciona una o varias categorías de la lista.",
    options=forecast_consolidado['Categoria'].unique()
)

# Aplicar los filtros al DataFrame
prediccion_filtrada = forecast_consolidado.copy()

if productos_seleccionados:
    prediccion_filtrada = prediccion_filtrada[prediccion_filtrada['Producto'].isin(productos_seleccionados)]

if categorias_seleccionadas:
    prediccion_filtrada = prediccion_filtrada[prediccion_filtrada['Categoria'].isin(categorias_seleccionadas)]

# Mostrar resultados
if not prediccion_filtrada.empty:
    columnas_para_mostrar = ['GTIN', 'Producto', 'Categoria', 'Campus', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024', 'Stock']
    st.dataframe(prediccion_filtrada[columnas_para_mostrar])
else:
    st.write("No se encontraron datos con los filtros seleccionados.")

#######

# --- Gráfico 1: Comparación de Stock vs Predicciones por Categoría ---
st.subheader("Comparación de Stock vs Predicciones por Categoría")
comparacion_categoria = forecast_pivot.groupby('Categoría')[['Stock', 'Septiembre 2024', 'Octubre 2024', 'Noviembre 2024']].sum().reset_index()
st.bar_chart(data=comparacion_categoria.set_index('Categoría'))
