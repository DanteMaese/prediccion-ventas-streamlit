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
