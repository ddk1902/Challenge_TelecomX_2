# TelecomX - Predicción de Churn de Clientes

> **Análisis de Machine Learning para predecir la cancelación de servicios (churn)**  
> Parte 2 del proyecto: Modelado, Evaluación e Interpretación

---

## Propósito del Análisis

El objetivo principal de este proyecto es **predecir el churn (abandono de clientes)** en TelecomX, una empresa de telecomunicaciones, utilizando técnicas de **Machine Learning** y análisis de datos.

A través de un conjunto de datos que incluye información demográfica, servicios contratados, métodos de pago y cargos, se busca identificar patrones que anteceden la cancelación del servicio. La capacidad de anticipar el churn permite a la empresa implementar **estrategias proactivas de retención**, mejorar la experiencia del cliente y reducir pérdidas de ingresos.

Este análisis forma parte de una solución integral de inteligencia predictiva para la toma de decisiones estratégicas.

---

##  Estructura del Proyecto
TelecomX-Churn-Prediction/

│

├── TelecomX_Data.json # Datos originales en formato JSON (anidados)

├── telecomx_data_clean.csv # Dataset procesado y aplanado (listo para ML)

├── notebook_churn_prediction.ipynb # Cuaderno principal con todo el flujo de análisis

│

├── visualizaciones/ # Carpeta con gráficos exportados

│ ├── balanceo_clases.png

│ ├── correlaciones.png

│ ├── importancia_variables.png

│ ├── curvas_roc.png

│ └── resumen_ejecutivo.png

│

├── modelos/ # Modelos entrenados

│ └── best_model.pkl

│

└── README.md # Documentación del proyecto

---

## Proceso de Preparación de Datos

### 1. **Clasificación de Variables**

Las variables fueron clasificadas automáticamente en tres categorías:

- **Binarias (Yes/No)**:  
  `PhoneService`, `MultipleLines`, `OnlineSecurity`, `TechSupport`, etc.

- **Categóricas Múltiples**:  
  `InternetService`, `Contract`, `PaymentMethod`, `gender`, etc.

- **Numéricas**:  
  `tenure`, `MonthlyCharges`, `TotalCharges`.

### 2. **Codificación de Variables**

- **Variables binarias**:  
  Mapeo: `Yes = 1`, `No = 0`.  
  Casos especiales (ej. `No internet service`) mapeados a `-1`.

- **Variables categóricas múltiples**:  
  Codificación **One-Hot Encoding** con `pd.get_dummies()`.

- **Variable objetivo (`Churn`)**:  
  Codificada como `No = 0`, `Yes = 1`.

### 3. **Normalización**

- **Datos numéricos**:  
  Estandarizados usando `StandardScaler` para modelos sensibles a escala (Regresión Logística, SVM, KNN).

- **Datos categóricos**:  
  No normalizados (usados en modelos basados en árboles).

### 4. **División de Datos**

- **70% entrenamiento**, **30% prueba**  
- Estratificación por la variable `Churn` para mantener la proporción de clases.  
- Aplicación de **SMOTE** (opcional) para balancear clases si el ratio de desbalance > 2.0.

---

##  Justificación de Decisiones de Modelización

| Decisión | Justificación |
|--------|---------------|
| **Uso de múltiples modelos** | Comparar enfoques distintos: lineales, basados en árboles y basados en distancia. |
| **Separación de datos escalados vs. sin escalar** | Aprovechar el rendimiento óptimo de cada modelo según su sensibilidad a la escala. |
| **Pesos de clase (`class_weight='balanced'`)** | Manejar el desbalance de clases sin sobre-muestreo agresivo. |
| **Evaluación con F1-Score** | Métrica ideal para problemas desbalanceados (equilibrio entre precision y recall). |
| **Permutation Importance** | Permitir interpretación de modelos que no tienen `feature_importances_` (ej. SVM). |

---

## Ejemplos de Gráficos e Insights del EDA

### Insights Clave

- **Los clientes con contrato mes a mes tienen 3.5x más probabilidad de churn.**
- **Clientes con fibra óptica presentan tasas de churn más altas** (posiblemente por problemas de servicio).
- **El tiempo como cliente (`tenure`) es la variable más predictiva**: a menor tenure, mayor riesgo.
- **Pagos con cheque electrónico están fuertemente asociados al churn.**

###  Visualizaciones Principales

| Gráfico | Descripción |
|-------|-------------|
| **Distribución de Churn** | Muestra desbalance moderado (73% No Churn, 27% Churn). |
| **Matriz de Correlación** | Destaca variables como `tenure`, `Contract`, y `MonthlyCharges` como altamente correlacionadas con churn. |
| **Top Features (Ranking Consolidado)** | Muestra las variables más consistentes entre múltiples modelos. |
| **Curvas ROC** | Compara el rendimiento de todos los modelos (AUC > 0.8 para los mejores). |
| **Matrices de Confusión** | Muestra verdaderos positivos y falsos negativos clave para estrategias de retención. |

---

## Instrucciones para Ejecutar el Cuaderno

### 1. **Requisitos del Sistema**

Asegúrate de tener instalado **Python 3.8+** y las siguientes bibliotecas:

pandas

numpy

matplotlib

seaborn

scikit-learn

imbalanced-learn

2. Ejecución del Cuaderno
   
Coloca el archivo TelecomX_Data.json en la misma carpeta del cuaderno.

Abre el cuaderno con Jupyter:

jupyter notebook notebook_churn_prediction.ipynb

Ejecuta todas las celdas en orden.

4. Carga de Datos Tratados (CSV)
   
Si deseas saltar la limpieza y usar el dataset ya procesado

df_final = pd.read_csv('telecomx_data_clean.csv')

Este CSV contiene todas las variables codificadas, normalizadas y listas para modelado.

## Resultado Final

El modelo Random Forest fue seleccionado como el mejor según el F1-Score, con un rendimiento robusto y buena interpretabilidad. Se proyecta un ROI positivo en el primer año al implementar un sistema de retención proactiva basado en este modelo.

## Contacto

Proyecto desarrollado como parte de un análisis de ciencia de datos aplicada.

Para consultas o colaboraciones: gomezdiego1902@gmail.com

