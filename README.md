
# life-expectancy-data

# 📊 Dashboard: Análisis de Competitividad vs. Consumo de Alcohol

Este proyecto implementa un **Dashboard interactivo** desarrollado en Python con **Streamlit** para analizar la relación entre la ingesta de alcohol per cápita y el estatus de competitividad de un país, utilizando un modelo de **Regresión Logística Binaria**.

El análisis se centra en países miembros de esquemas de integración económica específicos (CAN, MERCOSUR, T-MEC, UE) para el periodo **2008-2015**.

## 📝 Contexto de la Investigación

La **competitividad** se define como "el conjunto de instituciones, políticas y factores que determinan el nivel de productividad de un país". Operacionalmente, esta distinción se equipara con la clasificación de **País Desarrollado o En Desarrollo (Estatus)**.

El estudio busca determinar si el consumo de alcohol (etílico) afecta el estatus de los países, considerando que el abuso puede impactar la productividad nacional por enfermedades, accidentes o muertes asociadas entre los trabajadores.

**Objetivo General:**
Determinar si el consumo de alcohol es un predictor estadísticamente significativo de la clasificación de Estatus de los países miembros de la CAN, MERCOSUR, T-MEC y la Unión Europea.

**Variables en el Modelo:**
* **Variable Dependiente (Y):** `Estatus` (Binario: Developed / Developing)
* **Variables Predictoras (X):** * `Alcohol` (consumo per cápita en litros de alcohol puro)
    * `Mortalidad_Adultos` (tasa de mortalidad entre 15 y 60 años/1000 hab.)
    * `Escolaridad` (número de años de estudio)

***

## ⚙️ Estructura del Proyecto

El corazón del proyecto es el archivo `app.py`, que realiza las siguientes tareas:

1.  **Carga y Limpieza de Datos:** Filtra los datos del archivo `Life_Expectancy_Dataset.csv` por el rango de años (2008-2015) y los países de interés definidos.
2.  **Modelado:** Ajusta un modelo de Regresión Logística Binaria utilizando la librería `statsmodels` para predecir el `Estatus_Binario` (1 = Desarrollado).
3.  **Visualización:** Construye un dashboard interactivo utilizando **Streamlit** y **Plotly** para mostrar los resultados.

***

## 🔬 Resultados y Conclusiones Clave

### 1. Métricas de Clasificación

El modelo presenta una **Precisión (Accuracy)** alta, lo que indica su buen desempeño general:

| Métrica | Valor |
| :--- | :--- |
| **Precisión del Modelo (Accuracy)** | $0.89$ |
| **Verdaderos Positivos (VP)** | X (Países Developed clasificados correctamente) |
| **Falsos Negativos (FN)** | **0** (Países Developed clasificados incorrectamente) |
| **Falsos Positivos (FP)** | **38** (Países Developing clasificados incorrectamente) |

**Conclusión de la Matriz de Confusión:**
El modelo tiene una **Sensibilidad perfecta** para identificar a los países Desarrollados (**FN=0**). El principal error proviene de clasificar países En Desarrollo como Desarrollados (**FP=38**), sugiriendo que algunas naciones en desarrollo tienen un perfil (ej. alto consumo de alcohol o escolaridad) que imita al de un país desarrollado.

### 2. Interpretación de Odds Ratios (OR)

Los Odds Ratios (OR), calculados como $\text{e}^{\text{Coeficiente}}$, indican el cambio en las *Odds* de que un país sea clasificado como **Desarrollado** por cada unidad de aumento en la variable predictora.

| Variable | Odds Ratio ($\text{e}^{\text{Coef.}}$) | Interpretación |
| :--- | :--- | :--- |
| **Alcohol** | **> 1** | Un aumento en el consumo de alcohol **aumenta** las *Odds* de ser Desarrollado. |
| **Escolaridad** | **> 1** | Un aumento en la Escolaridad **aumenta fuertemente** las *Odds* de ser Desarrollado (coherente con competitividad). |
| **Mortalidad_Adultos** | **< 1** | Un aumento en la Mortalidad Adulta **disminuye** las *Odds* de ser Desarrollado (coherente con un desarrollo pobre). |

**Conclusión del Modelo:**
El **consumo de alcohol es un predictor positivo y significativo** del estatus. No implica causalidad, sino que los países clasificados como Desarrollados tienden a tener un consumo per cápita de alcohol más alto que los clasificados En Desarrollo.

***

 ## Link del despliegue

link: [https://life-expectancy-data-jaiynwwh7ynu7efbikrgdg.streamlit.app/]
