import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Inicio Parte 1

RUTA_ARCHIVO = "Ventas.xlsx"

st.title("TECStore - Detalle de ventas futuras y prescripción de inventarios")

@st.cache_data
def cargar_datos():
    """Carga y limpia los datos de Excel, excluyendo registros de 'Tecmilenio'."""
    df = pd.read_excel(RUTA_ARCHIVO)
    df = df[df['Empresa'] != 'Tecmilenio']  # Excluir registros de Tecmilenio
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

# --- Cargar y procesar datos iniciales ---
df, df_TS = cargar_datos()
monthly_df = procesar_datos(df_TS)

# --- Selector de Campus ---
# Obtener lista única de campus
lista_campus = sorted(df['Campus'].dropna().unique())

# Mostrar selector de campus
campus_seleccionado = st.selectbox(
    "Selecciona un campus para generar la predicción:",
    options=["Campus"] + lista_campus,
    index=0
)

# Validar selección de campus
if campus_seleccionado == "Campus":
    st.stop()

# Filtrar datos según el campus seleccionado
monthly_df = monthly_df[monthly_df['Campus'] == campus_seleccionado]

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

# Final Parte 1

# Inicio Parte 2

# --- Cargar y procesar los datos usando las funciones cacheadas ---
forecast_df = generar_predicciones(monthly_df)

# Extraer las columnas únicas de GTIN, Producto y Categoría para el join
info_producto = df[['GTIN', 'Producto', 'Categoría']].drop_duplicates()

# Realizar el join para agregar Producto y Categoría a forecast_df
forecast_df = forecast_df.merge(info_producto, on='GTIN', how='left')

# Final Parte 2

# Inicio Parte 3

# Definir la ruta del archivo BD Stock
RUTA_STOCK = "Stock.xlsx"

# --- Cargar el archivo de stock y realizar el join ---
@st.cache_data
def cargar_stock():
    try:
        stock_df = pd.read_excel(RUTA_STOCK)
        stock_df['GTIN'] = pd.to_numeric(stock_df['GTIN'], errors='coerce').fillna(0).astype('int64')  # Asegurar que GTIN sea numérico
        return stock_df
    except FileNotFoundError:
        st.error(f"El archivo '{RUTA_STOCK}' no se encuentra. Verifica que esté en el repositorio.")
        return pd.DataFrame()

# Cargar los datos de stock
stock_df = cargar_stock()

# Consolidar las predicciones en columnas por fecha
forecast_consolidado = forecast_df.pivot_table(
    index=['GTIN', 'Producto', 'Categoría', 'Campus'],  # Incluir Producto y Categoría en el índice
    columns='Fecha',                                    # Las fechas se convierten en columnas
    values='Predicción de Unidades',                    # Usar las predicciones como valores
    aggfunc='sum'                                       # Sumar valores si hay duplicados
).reset_index()

# Renombrar columnas según las fechas mapeadas
fechas_mapeo = {
    pd.Timestamp('2024-09-30'): 'Pred. Sep 2024',
    pd.Timestamp('2024-10-31'): 'Pred. Oct 2024',
    pd.Timestamp('2024-11-30'): 'Pred. Nov 2024'
}
forecast_consolidado.rename(columns=fechas_mapeo, inplace=True)

# Asegurar que GTIN sea numérico en forecast_consolidado
forecast_consolidado['GTIN'] = pd.to_numeric(forecast_consolidado['GTIN'], errors='coerce').fillna(0).astype('int64')

# Realizar el join para agregar el stock
forecast_consolidado = forecast_consolidado.merge(
    stock_df[['GTIN', 'Stock']], 
    on='GTIN', 
    how='left'
)

# Asegurarse de que la columna 'Producto' no tenga valores nulos y convertir a string
forecast_consolidado['Producto'] = forecast_consolidado['Producto'].fillna("").astype(str)

# Crear el DataFrame inicial para aplicar filtros
df_filtrado = forecast_consolidado.copy()

# Filtros para Producto y Categoría
productos_seleccionados = st.multiselect(
    "Escribe o selecciona uno o varios productos:",
    options=sorted(df_filtrado['Producto'].unique())  # Ordenar productos alfabéticamente
)

categorias_seleccionadas = st.multiselect(
    "Escribe o selecciona una o varias categorías:",
    options=df_filtrado['Categoría'].unique()
)

# Aplicar los filtros al DataFrame consolidado
if productos_seleccionados:
    df_filtrado = df_filtrado[df_filtrado['Producto'].isin(productos_seleccionados)]

if categorias_seleccionadas:
    df_filtrado = df_filtrado[df_filtrado['Categoría'].isin(categorias_seleccionadas)]

# Seleccionar columnas relevantes
columnas_para_mostrar = ['GTIN', 'Producto', 'Categoría', 'Campus', 'Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024', 'Stock']

# Formatear y mostrar el DataFrame
if not df_filtrado.empty:
    st.subheader("Predicción Consolidada para los Filtros Seleccionados")
    
    # Convertir el GTIN a string para evitar formato numérico con comas
    df_filtrado['GTIN'] = df_filtrado['GTIN'].astype(str)
    
    # Formatear columnas de predicciones para mostrar dos decimales
    columnas_prediccion = ['Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024']
    for columna in columnas_prediccion:
        df_filtrado[columna] = df_filtrado[columna].map("{:.2f}".format)
    
    # Mostrar el DataFrame con formato mejorado
    st.dataframe(df_filtrado[columnas_para_mostrar], use_container_width=True)
else:
    st.warning("No se encontraron predicciones que coincidan con los filtros seleccionados.")

# Final Parte 3

# Inicio Parte 4

# -- Plot 1: Comparación Total Predicciones vs. Stock --
import plotly.graph_objects as go

if not df_filtrado.empty:
    # Calcular totales
    total_predicciones = df_filtrado[['Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024']].astype(float).sum().sum()
    total_stock = df_filtrado['Stock'].astype(float).sum()

    # Crear el gráfico de barras
    fig = go.Figure()

    # Barra de predicciones
    fig.add_trace(go.Bar(
        x=["Unidades Predichas"],
        y=[total_predicciones],
        text=[f"{total_predicciones:.2f}"],  # Texto dentro de la barra
        textposition='inside',
        name="Unidades Predichas",
        marker_color='blue'
    ))

    # Barra de stock
    fig.add_trace(go.Bar(
        x=["Stock Disponible"],
        y=[total_stock],
        text=[f"{total_stock:.2f}"],
        textposition='inside',
        name="Stock Disponible",
        marker_color='green'
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title="Comparación: Total Predicciones vs. Stock Disponible",
        xaxis_title="",
        yaxis_title="Unidades",
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        font=dict(size=12),
        height=300,
        margin=dict(t=50, b=50)
    )

    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Por favor, selecciona un producto o categoría para visualizar las predicciones.")

# --- Plot 2: Análisis de Liquidación con Métricas Adicionales ---
# Extraer las columnas necesarias del archivo original
df_adicional = df[['GTIN', 'Piezas', 'Precio Unitario', 'Costo Unitario']].copy()

# Calcular el total de piezas por GTIN
df_adicional = df_adicional.groupby('GTIN', as_index=False).agg(
    Piezas_Vendidas=('Piezas', 'sum'),
    Precio_Unitario=('Precio Unitario', 'first'),
    Costo_Unitario=('Costo Unitario', 'first')
)

# Asegurar que GTIN en ambos DataFrames sea del mismo tipo
df_filtrado['GTIN'] = df_filtrado['GTIN'].astype('int64')
df_adicional['GTIN'] = df_adicional['GTIN'].astype('int64')

# Merge para combinar la información adicional con df_filtrado
df_filtrado = df_filtrado.merge(df_adicional, on='GTIN', how='left')

# Reemplazar valores NaN en columnas relevantes
df_filtrado['Stock'] = df_filtrado['Stock'].fillna(0.00)

# Calcular el promedio mensual basado en predicciones y piezas vendidas (histórico)
df_filtrado['Promedio Mensual'] = (
    df_filtrado[['Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024']].astype(float).sum(axis=1) +
    df_filtrado['Piezas_Vendidas'].astype(float).fillna(0)
) / 12

# Excedente de Inventario Actual - Excedente = Stock - (Promedio Mensual * 6)
df_filtrado['Unidades para Rematar'] = (
    df_filtrado['Stock'].astype(float) - (df_filtrado['Promedio Mensual'] * 6)
).clip(lower=0)

# Precio de Remate por Unidad
df_filtrado['Precio de Remate por Unidad'] = (
    (df_filtrado['Precio_Unitario'] - df_filtrado['Costo_Unitario']) * 0.8
).clip(lower=0)

# Total a Generar por Remate
df_filtrado['Total a Generar por Remate'] = (
    df_filtrado['Unidades para Rematar'] * df_filtrado['Precio de Remate por Unidad']
)

# Formatear columnas para visualización
columnas_para_mostrar = ['GTIN', 'Producto', 'Categoría', 'Campus',
                         'Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024',
                         'Stock', 'Piezas_Vendidas', 'Precio_Unitario', 'Costo_Unitario',
                         'Promedio Mensual', 'Unidades para Rematar', 'Precio de Remate por Unidad', 'Total a Generar por Remate']

# Mostrar el DataFrame actualizado en Streamlit
if not df_filtrado.empty:
    st.subheader("Análisis de Liquidación con Métricas Adicionales")
    st.dataframe(df_filtrado[columnas_para_mostrar], use_container_width=True)
else:
    st.write("No se encontraron datos para los filtros seleccionados.")

# --- Gráfico de Análisis de Liquidación ---
productos_a_rematar = df_filtrado[df_filtrado['Unidades para Rematar'] > 0]

if not productos_a_rematar.empty:
    # Crear un DataFrame para el gráfico (solo piezas vendidas, stock y unidades para rematar)
    df_plot = productos_a_rematar[['Producto', 'Piezas_Vendidas', 'Stock', 'Unidades para Rematar']].copy()

    fig = px.bar(
        df_plot.melt(id_vars='Producto', value_vars=['Piezas_Vendidas', 'Stock', 'Unidades para Rematar']),
        x='Producto',
        y='value',
        color='variable',
        title="Análisis de Liquidación: Inventario y Predicciones",
        labels={'value': 'Unidades', 'variable': 'Métricas'},
        barmode='group',
        text_auto=True
    )

    fig.update_layout(
        xaxis_title="Productos",
        yaxis_title="Unidades",
        legend_title="Métricas",
        height=400,
        margin=dict(t=50, b=50),
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No se encontraron productos con exceso de stock para liquidar.")

# Final Parte 4
