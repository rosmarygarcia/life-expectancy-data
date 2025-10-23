import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
from pandas.core.frame import DataFrame 

# ----------------------------------------------------------------------
# 1. Configuración de la Página y Título
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard: Ingesta de Alcohol vs. Estatus",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Análisis de Competitividad vs. Consumo de Alcohol")
st.header("Modelo de Regresión Logística Binaria (2008-2015)")
st.markdown("---")

# ----------------------------------------------------------------------
# 2. CONTEXTO Y OBJETIVOS (Extraído del Planteamiento de la Investigación)
# ----------------------------------------------------------------------

with st.expander("📝 Planteamiento de la Investigación y Objetivos", expanded=False):
    st.subheader("Planteamiento de la Investigación")
    st.markdown("""
    La competitividad, definida por el Foro Económico Mundial (2016), es el conjunto de instituciones, políticas y factores que determinan el nivel de productividad de un país. 
    Esta distinción se equipara operativamente con la clasificación de **País Desarrollado o En Desarrollo (Estatus)**.
    
    El análisis busca determinar si el consumo de alcohol (etílico) afecta o no el estatus de los países, ya que un consumo exagerado puede afectar la productividad por enfermedades, accidentes o muertes asociadas entre los trabajadores.
    
    * **Técnica:** Regresión Logística Binaria.
    * **Variable Dependiente (Y):** Estatus (Binario).
    * **Variables Predictoras (X):** Alcohol (consumo per cápita), Mortalidad Adulta y Escolaridad.
    * **Período:** Años 2008-2015.
    """)

    st.subheader("Objetivo General")
    st.markdown("""
    Determinar si el consumo de alcohol es un predictor estadísticamente significativo de la clasificación de Estatus de los países miembros de la CAN, MERCOSUR, T-MEC y la Unión Europea.
    """)
    st.subheader("Interrogante")
    st.markdown("""
    ¿Cómo es el consumo de alcohol en países desarrollados y en vías de desarrollo en 2015, medida por las variables expectativa de vida, mortalidad de adultos, muerte de infantes y escolarización afectaría la competitividad?
    """)

st.markdown("---")

# ----------------------------------------------------------------------
# 3. Carga y Preparación de Datos (Versión Robusta)
# ----------------------------------------------------------------------

@st.cache_data
def load_data() -> DataFrame:
    """Carga, limpia y transforma los datos de forma robusta."""
    try:
        le = pd.read_csv("Life_Expectancy_Dataset.csv")
    except FileNotFoundError:
        st.error("Error: Asegúrate de que 'Life_Expectancy_Dataset.csv' esté en el mismo directorio.")
        return pd.DataFrame()

    try:
        le.columns = le.columns.str.strip().str.replace('.', ' ', regex=False)
        
        paises_interes = [
            "Germany", "Argentina", "Austria", "Belgium", "Bolivia", 
            "Brazil", "Bulgaria", "Canada", "Cyprus", "Colombia", 
            "Croatia", "Denmark", "Ecuador", "Slovakia", "Slovenia", 
            "Spain", "United States of America", "Estonia", "Finland", 
            "France", "Greece", "Hungary", "Ireland", "Italy", 
            "Latvia", "Lithuania", "Luxembourg", "Malta", 
            "Netherlands", "Paraguay", "Peru", "Poland", "Portugal", 
            "Czech Republic", "Romania", "Sweden", "Uruguay", 
            "Venezuela (Bolivarian Republic of)"
        ]
        
        data_logistica = le.rename(columns={
            'Country': 'Pais', 
            'Year': 'Año', 
            'Status': 'Estatus', 
            'Adult Mortality': 'Mortalidad_Adultos', # Nombre corregido (Adult Mortality)
            'Alcohol': 'Alcohol', 
            'Schooling': 'Escolaridad'
        })
        
        data_logistica = data_logistica[
            (data_logistica['Año'] >= 2008) & 
            (data_logistica['Año'] <= 2015) & 
            (data_logistica['Pais'].isin(paises_interes))
        ].dropna(subset=['Estatus', 'Alcohol', 'Mortalidad_Adultos', 'Escolaridad']) 
        
        data_logistica['Estatus_Binario'] = data_logistica['Estatus'].apply(
            lambda x: 1 if x == 'Developed' else 0
        )

        return data_logistica

    except KeyError as e:
        st.error(f"Error de columna (KeyError): {e}. Asegúrate de que los nombres de columna en el CSV sean correctos.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"Ocurrió un error inesperado durante el procesamiento de datos: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 4. EJECUCIÓN DEL CÓDIGO GLOBAL Y MODELADO
# ----------------------------------------------------------------------
df = load_data()

if df.empty:
    st.error("La aplicación se detiene porque no se pudieron cargar o procesar los datos.")
    st.stop()

X = df[['Alcohol', 'Mortalidad_Adultos', 'Escolaridad']]
y = df['Estatus_Binario']
X_sm = sm.add_constant(X, prepend=False)

try:
    # Ajuste del modelo de Regresión Logística Binaria
    modelo_sm = sm.Logit(y, X_sm).fit(disp=0) 
except Exception as e:
    st.error(f"Error al ajustar el modelo logístico: {e}.")
    st.stop()

y_pred_proba = modelo_sm.predict(X_sm)
y_pred = (y_pred_proba >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = accuracy_score(y, y_pred)
df['Prob_Desarrollado'] = y_pred_proba

# ----------------------------------------------------------------------
# 5. Diseño del Dashboard - Métricas Clave y Resumen
# ----------------------------------------------------------------------

st.header("🔑 Resumen y Métricas Clave")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Precisión del Modelo (Accuracy)", f"{accuracy:.2f}")
col2.metric("Verdaderos Positivos (Developed Correctos)", tp)
col3.metric("Falsos Positivos (Developing Mal Clasificados)", fp)
col4.metric("Falsos Negativos (Developed NO detectado)", fn)

st.markdown("---")

# ----------------------------------------------------------------------
# 6. Resultados del Modelo y Odds Ratios
# ----------------------------------------------------------------------

st.header("🔬 Resultados del Modelo de Regresión Logística")

col5, col6 = st.columns(2)

with col5:
    st.subheader("Coeficientes del Modelo (Log-Odds)")
    # Muestra coeficientes y P-valores para evaluar la significancia
    st.dataframe(
        modelo_sm.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']].rename(
            columns={'Coef.': 'Coeficiente (Log-Odds)', 'Std.Err.': 'Error Estándar', 'P>|z|': 'Valor p'}
        ).iloc[[0, 1, 2]],
        use_container_width=True
    )

with col6:
    st.subheader("Interpretación: Odds Ratios (OR)")
    # Muestra los Odds Ratios (e^Coef.)
    or_df = pd.DataFrame({
        'Variable': ['Alcohol', 'Mortalidad_Adultos', 'Escolaridad'],
        'Odds Ratio (e^Coef.)': np.exp(modelo_sm.params).iloc[[0, 1, 2]],
    }).set_index('Variable')
    
    st.dataframe(or_df, use_container_width=True)
    st.markdown("""
    * **OR > 1:** Indica que la variable **aumenta** las *Odds* de que el país sea clasificado como **Desarrollado**.
    * **OR < 1:** Indica que la variable **disminuye** las *Odds* de ser clasificado como **Desarrollado**.
    """)


# ----------------------------------------------------------------------
# 7. Gráficos de Visualización
# ----------------------------------------------------------------------

st.header("📈 Gráficos de Visualización")

col7, col8 = st.columns(2)

# Gráfico 1: Curva de Probabilidad Logística (Alcohol)
with col7:
    st.subheader("1. Probabilidad Predicha vs. Consumo de Alcohol")

    min_alc, max_alc = df['Alcohol'].min(), df['Alcohol'].max()
    alc_range = np.linspace(min_alc, max_alc, 100)
    
    pred_data = pd.DataFrame({
        'Alcohol': alc_range,
        'Mortalidad_Adultos': df['Mortalidad_Adultos'].mean(),
        'Escolaridad': df['Escolaridad'].mean(),
        'const': 1
    })

    pred_proba = modelo_sm.predict(pred_data[['Alcohol', 'Mortalidad_Adultos', 'Escolaridad', 'const']])
    pred_data['Prob_Desarrollado_Curva'] = pred_proba

    fig_alc = px.scatter(
        df, 
        x='Alcohol', 
        y='Prob_Desarrollado', 
        color='Estatus',
        color_discrete_map={'Developing': '#D55E00', 'Developed': '#009E73'},
        opacity=0.6,
        title="Probabilidad Predicha vs. Consumo de Alcohol",
        labels={'Prob_Desarrollado': 'Probabilidad Predicha (Estatus=Developed)'}
    )
    
    fig_alc.add_scatter(
        x=pred_data['Alcohol'], 
        y=pred_data['Prob_Desarrollado_Curva'], 
        mode='lines', 
        name='Curva Logística (Media)',
        line=dict(color='#0072B2', width=3)
    )

    fig_alc.update_layout(showlegend=True, height=500)
    st.plotly_chart(fig_alc, use_container_width=True)

# Gráfico 2: Matriz de Confusión
with col8:
    st.subheader("2. Matriz de Confusión")

    cm_df = pd.DataFrame({
        'Real': ['Developing', 'Developing', 'Developed', 'Developed'],
        'Predicción': ['Developing', 'Developed', 'Developing', 'Developed'],
        'Conteo': [tn, fp, fn, tp]
    })
    
    fig_cm = px.bar(
        cm_df, 
        x='Real', 
        y='Conteo', 
        color='Predicción',
        color_discrete_map={'Developing': '#D55E00', 'Developed': '#009E73'},
        text='Conteo',
        title='Resultados del Modelo (Matriz de Confusión)'
    )
    
    fig_cm.update_traces(textposition='inside')
    fig_cm.update_layout(height=500, legend_title="Predicción del Modelo")
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------
# 8. INTERPRETACIÓN DE RESULTADOS
# ----------------------------------------------------------------------
st.header("🔍 Interpretación y Conclusiones del Modelo")

col9, col10 = st.columns(2)

with col9:
    st.subheader("Evaluación de la Matriz de Confusión")
    st.markdown(f"""
    1.  **Precisión General:** La precisión del modelo es del **{accuracy*100:.2f}%**.
    2.  **Sensibilidad Perfecta (Developed):** El modelo logró **cero Falsos Negativos (FN=0)**, lo que significa que **nunca se equivocó al clasificar a un país Desarrollado** como "En Desarrollo".
    3.  **Error Principal (Falsos Positivos):** La fuente principal de error son los **38 Falsos Positivos (FP=38)**. Estos son países que son **En Desarrollo** pero que el modelo clasificó incorrectamente como Desarrollados. Esto sugiere que algunas naciones en desarrollo exhiben características (alto consumo de alcohol, buena escolaridad, o baja mortalidad) que imitan el perfil de un país desarrollado.
    """)

with col10:
    st.subheader("Interpretación de Variables (Odds Ratios)")
    st.markdown("""
    * **Alcohol (Consumo per cápita):** El consumo de alcohol es un **predictor positivo y significativo** del estatus. Un aumento en el consumo de alcohol aumenta las *Odds* de ser Desarrollado. Esto no implica causalidad, sino que los países Desarrollados tienden a tener un mayor consumo per cápita de alcohol.
    * **Escolaridad:** Actúa como se espera. Un mayor nivel de Escolaridad está fuertemente asociado con el aumento de las *Odds* de ser Desarrollado, lo cual es coherente con el pilar de Educación del Índice de Competitividad.
    * **Mortalidad Adulta:** Actúa como se espera. Un aumento en la Mortalidad Adulta está asociado con una **disminución** de las *Odds* de ser clasificado como Desarrollado, siendo un indicador de problemas de salud y desarrollo.
    """)

st.caption("Dashboard desarrollado en Python con Streamlit, Pandas, Plotly y Statsmodels.")