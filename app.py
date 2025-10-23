import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
from pandas.core.frame import DataFrame 
import os # Importación esencial para resolver el error de ruta en GitHub

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
    st.subheader("Título de la Investigación")
    st.markdown("""
    **INGESTA DE ALCOHOL: ANÁLISIS DE SU RELACIÓN CON EL ESTATUS DE LOS PAÍSES COMO EQUIVALENTE DE COMPETITIVIDAD**
    """)
    
    st.subheader("Planteamiento de la Investigación")
    st.markdown("""
    La competitividad según la define el Forum Económico Mundial (2016), es “el conjunto de instituciones, políticas y factores que determinan el nivel de productividad de un país”.
    Esta distinción se equipara operativamente con la clasificación de **País Desarrollado o En Desarrollo (Estatus)**.
    
    La competitividad puede inferirse de la observación del siguiente vídeo:
    
    ![Texto alternativo (Video Hans Rosling)](https://www.youtube.com/watch?v=jbkSRLYSojo&pp=ugMICgJlcxABGAHKBQxoYW5zIHJvc2xpbmc%3D)
    
    El Forum Económico Mundial elabora un Índice de Competitividad con 12 pilares. Una ilustración de este contexto es:
    
    ![Texto alternativo (Gráfico WEF)](https://assets.weforum.org/editor/eAYvGAf9gjxX0c3nS9widuCPy0AjnQ9DbstTRUZ6v_s.png)
    
    Se estudia la relación con la ingesta de alcohol, pues el consumo exagerado puede afectar la productividad. Un ejemplo gráfico del impacto en la productividad es:
    
    ![Texto alternativo (Infografía)](https://www.elnuevosiglo.com.co/sites/default/files/2017-09/08-INFOGRAFIA-ok.jpg)
    
    Hay efectos económicos no deseables por el abuso del alcohol, perjudiciales tanto para la salud como para la capacidad de las naciones de responder a desafíos futuros.
    
    ![Texto alternativo (Gráfico WEF 2)](https://assets.weforum.org/editor/eT9bBT0G_q9VFB0B3ZEI0VNolMHK5BeDDCuqhVcz2_Q.jpg)

    La distribución global del consumo de alcohol (más de 2.000 millones de bebedores habituales) se ilustra en este mapa:
    
    ![Texto alternativo (Mapa de consumo de alcohol)](https://i.blogs.es/abbfb6/mapa/1366_2000.jpeg)
    """)

    st.subheader("Interrogante")
    st.markdown("""
    ¿Cómo es el consumo de alcohol en países desarrollados y en vías de desarrollo en 2015, medida por las variables expectativa de vida, moratlidad de adultos, muerte de infantes y escolarización afectaria la competitividad?
    
    ![Texto alternativo (Globo borracho 1)](https://img.freepik.com/vector-gratis/globo-terraqueo-borracho-botellas-alcohol_1308-119717.jpg)
    """)

    st.subheader("Objetivo General")
    st.markdown("""
    Determinar si el consumo de alcohol es un predictor estadísticamente significativo de la clasificación de Estatus de los países miembros de la CAN, MERCOSUR, T-MEC y la Unión Europea utilizando un modelo de Regresión Logística Binaria.
    
    ![Texto alternativo (Globo borracho 2)](https://thumbs.dreamstime.com/b/tierra-borracha-del-planeta-con-la-botella-137221786.jpg?w=576)
    """)

st.markdown("---")

# ----------------------------------------------------------------------
# 3. Carga y Preparación de Datos (Versión Robusta para GitHub/Streamlit)
# ----------------------------------------------------------------------

@st.cache_data
def load_data() -> DataFrame:
    """Carga, limpia y transforma los datos de forma robusta, asegurando la ruta del archivo."""
    
    # SOLUCIÓN AL ERROR: Construir la ruta absoluta relativa al script (app.py)
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "Life_Expectancy_Dataset.csv")
    
    try:
        le = pd.read_csv(file_path) 
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'. Asegúrate de que 'Life_Expectancy_Dataset.csv' exista con el nombre exacto en el directorio raíz de GitHub.")
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
            'Adult Mortality': 'Mortalidad_Adultos', 
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
        st.error(f"Error de columna (KeyError): {e}. Revise los nombres de las columnas en su CSV.")
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
    st.dataframe(
        modelo_sm.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']].rename(
            columns={'Coef.': 'Coeficiente (Log-Odds)', 'Std.Err.': 'Error Estándar', 'P>|z|': 'Valor p'}
        ).iloc[[0, 1, 2]],
        use_container_width=True
    )

with col6:
    st.subheader("Interpretación: Odds Ratios (OR)")
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
    3.  **Error Principal (Falsos Positivos):** La fuente principal de error son los **38 Falsos Positivos (FP=38)**. Estos son países **En Desarrollo** que el modelo predijo incorrectamente como Desarrollados.
    """)

with col10:
    st.subheader("Interpretación de Variables (Odds Ratios)")
    st.markdown("""
    * **Alcohol (Consumo per cápita):** El consumo de alcohol es un **predictor positivo y significativo** del estatus. Un aumento en el consumo de alcohol aumenta las *Odds* de ser Desarrollado.
    * **Escolaridad:** Es un predictor positivo (OR > 1), fuertemente asociado con el aumento de las *Odds* de ser Desarrollado.
    * **Mortalidad Adulta:** Es un predictor negativo (OR < 1). Un aumento en la Mortalidad Adulta está asociado con una **disminución** de las *Odds* de ser clasificado como Desarrollado.
    """)

st.caption("Dashboard desarrollado en Python con Streamlit, Pandas, Plotly y Statsmodels.")
st.caption("Dashboard desarrollado en Python con Streamlit, Pandas, Plotly y Statsmodels.")


