import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Inicio Parte 1

# Definir la ruta del archivo como variable para facilidad de ajuste
RUTA_ARCHIVO = "Ventas.xlsx"

# Título de Streamlit
st.title("TECStore - Detalle de ventas futuras y prescripción de inventarios")

# --- Funciones de procesamiento con caché ---
@st.cache_data
def cargar_datos():
    """Carga y limpia los datos de Excel, excluyendo registros de 'Tecmilenio'."""
    df = pd.read_excel(RUTA_ARCHIVO)
    print("Columnas disponibles en el archivo:", df.columns)  # Para verificar nombres de columnas
    df = df[df['Empresa'] != 'Tecmilenio']
    df = df[df['Campus'] == 'Monterrey']
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
RUTA_STOCK = "Stock.xlsx"

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

# -- Total Predicciones vs. Stock Disponible

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

# --- Reglas de Negocio

# Extraer las columnas necesarias del archivo original
df_adicional = df[['GTIN', 'Piezas', 'Precio Unitario', 'Costo Unitario']].copy()

# Calcular el total de piezas por GTIN
df_adicional = df_adicional.groupby('GTIN', as_index=False).agg(
    Piezas_Vendidas=('Piezas', 'sum'),
    Precio_Unitario=('Precio Unitario', 'first'),  # Asume que el precio unitario es constante para cada GTIN
    Costo_Unitario=('Costo Unitario', 'first')  # Asume que el costo unitario es constante para cada GTIN
)

# Asegurarnos de que GTIN en ambos DataFrames sea del mismo tipo
df_filtrado['GTIN'] = df_filtrado['GTIN'].astype('int64')
df_adicional['GTIN'] = df_adicional['GTIN'].astype('int64')

# Merge para combinar la información adicional con df_filtrado
df_filtrado = df_filtrado.merge(df_adicional, on='GTIN', how='left')

# Reemplazar valores NaN en columnas relevantes
df_filtrado['Stock'] = df_filtrado['Stock'].fillna(0.00)

##### Cálculo basado en las Reglas de Negocio ######

# Convertir las columnas a float
df_filtrado[['Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024']] = (
    df_filtrado[['Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024']]
    .apply(pd.to_numeric, errors='coerce')  # Convertir a float; valores inválidos serán NaN
    .fillna(0)  # Rellenar NaN con 0
)

# Crear la columna Suma Predicciones
df_filtrado['Suma Predicciones'] = (
    df_filtrado['Pred. Sep 2024'] +
    df_filtrado['Pred. Oct 2024'] +
    df_filtrado['Pred. Nov 2024']
).fillna(0)

# Mostrar la tabla con la columna Suma Predicciones incluida
st.subheader("Tabla de Inventario con Predicciones y Estado")
columnas_para_mostrar = [
    'GTIN', 'Producto', 'Categoría', 'Campus', 
    'Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024',
    'Suma Predicciones', 'Stock'
]

# Inicializar columnas
df_filtrado['Estado Inventario'] = None
df_filtrado['Acción Recomendada'] = None

# Asegurar que 'Stock' y 'Suma Predicciones' sean numéricos
df_filtrado['Stock'] = pd.to_numeric(df_filtrado['Stock'], errors='coerce').fillna(0)
df_filtrado['Suma Predicciones'] = pd.to_numeric(df_filtrado['Suma Predicciones'], errors='coerce').fillna(0)

# Aplicar la condición SAFE ZONE
df_filtrado.loc[
    (df_filtrado['Stock'] >= 1.1 * df_filtrado['Suma Predicciones']) & 
    (df_filtrado['Stock'] <= 1.3 * df_filtrado['Suma Predicciones']),
    ['Estado Inventario', 'Acción Recomendada']
] = ["SAFE ZONE", "Inventario correcto"]



## Tabla ##

# Mostrar DataFrame completo en Streamlit
st.subheader("Tabla Completa de Inventario")
st.dataframe(df_filtrado[columnas_para_mostrar], use_container_width=True)

# Filtrar y mostrar solo las filas con 'SAFE ZONE'
safe_zone_df = df_filtrado[df_filtrado['Estado Inventario'] == "SAFE ZONE"]

if not safe_zone_df.empty:
    st.subheader("Filas en SAFE ZONE")
    st.dataframe(safe_zone_df[columnas_para_mostrar], use_container_width=True)
else:
    st.write("No hay filas en SAFE ZONE.")

# # COMPRA
# compra_condicion = df_filtrado['Stock'] < 1.1 * df_filtrado['Suma Predicciones']

# # Calcular cuántas piezas comprar, considerando la condición especial para stock cero
# piezas_a_comprar = (
#     (1.1 * df_filtrado['Suma Predicciones'] - df_filtrado['Stock']).clip(lower=0)
#     .where(df_filtrado['Stock'] != 0, df_filtrado['Suma Predicciones'])
# )

# # Asignar estado y acción recomendada para la condición de compra
# df_filtrado.loc[compra_condicion, 'Estado Inventario'] = "COMPRA"
# df_filtrado.loc[compra_condicion, 'Acción Recomendada'] = (
#     "Compra " + piezas_a_comprar[compra_condicion].round(2).astype(str) + " piezas"
# )

# # VENDE
# vende_condicion = df_filtrado['Stock'] > 1.3 * df_filtrado['Suma Predicciones']
# piezas_a_rematar = (df_filtrado['Stock'] - 1.3 * df_filtrado['Suma Predicciones']).clip(lower=0)
# df_filtrado.loc[vende_condicion, 'Estado Inventario'] = "VENDE"
# df_filtrado.loc[vende_condicion, 'Acción Recomendada'] = (
#     "Remata " + piezas_a_rematar[vende_condicion].round(2).astype(str) + " piezas"
# )

# ##### Fin de Cálculo basado en las Reglas de Negocio ######

# # Mostrar el DataFrame actualizado
# columnas_para_mostrar = ['GTIN', 'Producto', 'Categoría', 'Campus',
#                          'Pred. Sep 2024', 'Pred. Oct 2024', 'Pred. Nov 2024',
#                          'Stock', 'Estado Inventario', 'Acción Recomendada']

# if not df_filtrado.empty:
#     st.subheader("Análisis de Inventario con Reglas de Negocio")
#     st.dataframe(df_filtrado[columnas_para_mostrar], use_container_width=True)
# else:
#     st.write("No se encontraron datos para los filtros seleccionados.")
