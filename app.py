import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Inicio Parte 1

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

# Final Parte 1

# Inicio Parte 2

# --- Cargar y procesar los datos usando las funciones cacheadas ---
df, df_TS = cargar_datos()
monthly_df = procesar_datos(df_TS)
forecast_df = generar_predicciones(monthly_df)

# Extraer las columnas únicas de GTIN, Producto y Categoría para el join
info_producto = df[['GTIN', 'Producto', 'Categoría']].drop_duplicates()

# Realizar el join para agregar Producto y Categoría a forecast_df
forecast_df = forecast_df.merge(info_producto, on='GTIN', how='left')

# Final Parte 2

# Inicio Parte 3

# Definir la ruta del archivo BD Stock
RUTA_STOCK = "BD Stock.xlsx"

# --- Cargar el archivo de stock y realizar el join ---
@st.cache_data
def cargar_stock():
    """Carga los datos de stock desde el archivo BD Stock.xlsx."""
    try:
        stock_df = pd.read_excel(RUTA_STOCK)  # Usar la ruta definida para el archivo de stock
        stock_df['GTIN'] = stock_df['GTIN'].astype('int64')  # Convertir GTIN a int
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

# Asegurarse de que GTIN en ambos DataFrames esté en el mismo formato
forecast_consolidado['GTIN'] = forecast_consolidado['GTIN'].astype('int64')
stock_df['GTIN'] = stock_df['GTIN'].astype('int64')

# Realizar el join para agregar el stock
forecast_consolidado = forecast_consolidado.merge(stock_df[['GTIN', 'Stock']], on='GTIN', how='left')

# Mostrar resultados en Streamlit
st.title("Predicción Consolidada de Ventas - Campus MTY")

# Asegurarse de que la columna 'Producto' no tenga valores nulos y convertir a string
forecast_consolidado['Producto'] = forecast_consolidado['Producto'].fillna("").astype(str)

# Filtros para Producto y Categoría
productos_seleccionados = st.multiselect(
    "Escribe o selecciona uno o varios productos:",
    options=sorted(forecast_consolidado['Producto'].unique())  # Ordenar productos alfabéticamente
)

categorias_seleccionadas = st.multiselect(
    "Escribe o selecciona una o varias categorías:",
    options=forecast_consolidado['Categoría'].unique()
)

# Aplicar los filtros al DataFrame consolidado
df_filtrado = forecast_consolidado.copy()

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
    st.write("No se encontraron predicciones que coincidan con los filtros seleccionados.")
    
# Final Parte 3

# Inicio Parte 4

# -- Plot 1

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
        textposition='inside',  # Mostrar los valores dentro de la barra
        name="Unidades Predichas",
        marker_color='blue'  # Color de la barra
    ))

    # Barra de stock
    fig.add_trace(go.Bar(
        x=["Stock Disponible"],
        y=[total_stock],
        text=[f"{total_stock:.2f}"],
        textposition='inside',  # Mostrar los valores dentro de la barra
        name="Stock Disponible",
        marker_color='green'
    ))

    # Configurar el diseño del gráfico
    fig.update_layout(
        title={
            'text': "Comparación: Total Predicciones vs. Stock Disponible",
            'y': 0.9,  # Ubicación vertical del título
            'x': 0.5,  # Centrar el título horizontalmente
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="",  # No se requiere título en el eje X
        yaxis_title="Unidades",
        barmode='group',  # Mostrar las barras agrupadas
        legend=dict(
            orientation="h",  # Leyenda horizontal
            yanchor="bottom",
            y=-0.3,  # Mover la leyenda debajo del gráfico
            xanchor="center",
            x=0.5  # Centrar la leyenda
        ),
        font=dict(size=12),
        height=300,  # Altura ajustada
        margin=dict(t=50, b=50)  # Reducir márgenes para que quepa mejor
    )

    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Por favor, selecciona un producto o categoría para visualizar las predicciones.")

# -- Plot 2: Análisis de Liquidación

# Enriquecer el DataFrame con datos adicionales
df_adicional = df[['GTIN', 'Costo Unitario', 'Piezas']].copy()

# Calcular el promedio mensual de ventas por GTIN
df_adicional['Promedio Mensual'] = df_adicional.groupby('GTIN')['Piezas'].transform('mean')

# Seleccionar las columnas necesarias y eliminar duplicados
df_adicional = df_adicional[['GTIN', 'Promedio Mensual', 'Costo Unitario']].drop_duplicates()

# Asegurar que GTIN en ambos DataFrames sea del mismo tipo
df_filtrado['GTIN'] = df_filtrado['GTIN'].astype('int64')
df_adicional['GTIN'] = df_adicional['GTIN'].astype('int64')

# Merge con el DataFrame Filtrado
df_filtrado = df_filtrado.merge(df_adicional, on='GTIN', how='left')

# Calcular el Precio de Remate (20% arriba del costo unitario)
df_filtrado['Precio de Remate'] = df_filtrado['Costo Unitario'] * 1.2

# Calcular el Total de Predicciones
df_filtrado['Total Predicciones'] = df_filtrado[['Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024']].astype(float).sum(axis=1)

# Calcular el exceso de stock según la regla
df_filtrado['Exceso de Stock'] = df_filtrado['Stock'] > (df_filtrado['Total Predicciones'] * 1.6)

# Filtrar productos con exceso de stock
productos_a_rematar = df_filtrado[df_filtrado['Exceso de Stock']]

if not productos_a_rematar.empty:
    # Gráfico de Total Predicciones vs. Stock
    fig = px.bar(
        productos_a_rematar,
        x='Producto',
        y=['Total Predicciones', 'Stock'],
        title="Análisis de Liquidación: Productos con Exceso de Stock",
        labels={'value': 'Unidades', 'variable': 'Concepto'},
        barmode='group',
        text_auto=True,  # Mostrar los valores encima de las barras
        color_discrete_map={
            'Total Predicciones': 'blue',
            'Stock': 'green'
        }
    )
    fig.update_layout(
        xaxis_title="Productos",
        yaxis_title="Unidades",
        legend_title="Concepto",
        height=500,  # Ajustar la altura para mejorar la visualización
        margin=dict(t=50, b=50),
        font=dict(size=12)  # Tamaño de fuente consistente
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla Complementaria
    st.subheader("Detalles de Productos a Rematar")
    columnas_tabla = ['GTIN', 'Producto', 'Stock', 'Promedio Mensual', 'Costo Unitario', 'Precio de Remate']
    st.dataframe(productos_a_rematar[columnas_tabla].style.format({
        'Costo Unitario': '${:.2f}',
        'Precio de Remate': '${:.2f}',
        'Promedio Mensual': '{:.2f}',
        'Stock': '{:.0f}'
    }), use_container_width=True)
else:
    st.write("No se encontraron productos con exceso de stock para liquidar.")


# Final Parte 4
