import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # Importamos SVM en lugar de Random Forest

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================================
# 1. CARGA DE DATOS Y LIMPIEZA BÁSICA
# =====================================================================
print("--- 1. Cargando datos ---")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y_train_raw = train['Churn'].map({'Yes': 1, 'No': 0})

test_ids = test['customerID']

# =====================================================================
# 2. INGENIERÍA DE VARIABLES
# =====================================================================
print("--- 2. Creando nuevas variables ---")

def engineer_features(df):

    df_new = df.copy()

    # -------------------------------------------------------------
    # Forzar variables numéricas
    # -------------------------------------------------------------
    for col in ['TotalCharges', 'tenure', 'MonthlyCharges']:
        df_new[col] = pd.to_numeric(
            df_new[col],
            errors='coerce'
        ).fillna(0)

    # -------------------------------------------------------------
    # Variables de servicios
    # -------------------------------------------------------------
    servicios = [
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    # Número total de servicios
    df_new['num_servicios'] = df_new[servicios].apply(
        lambda x: (x == 'Yes').sum(),
        axis=1
    )

    # -------------------------------------------------------------
    # Variables financieras
    # -------------------------------------------------------------
    df_new['gasto_promedio_real'] = (
        df_new['TotalCharges'] /
        df_new['tenure'].clip(lower=1)
    )

    df_new['costo_por_servicio'] = (
        df_new['MonthlyCharges'] /
        df_new['num_servicios'].clip(lower=1)
    )

    # -------------------------------------------------------------
    # Variables de comportamiento
    # -------------------------------------------------------------
    df_new['cliente_nuevo'] = (
        df_new['tenure'] <= 3
    ).astype(int)

    # Contrato mensual
    df_new['contrato_mensual'] = (
        df_new['Contract'] == 'Month-to-month'
    ).astype(int)

    # Fibra sin soporte
    df_new['fibra_sin_soporte'] = (
        (
            (df_new['InternetService'] == 'Fiber optic') &
            (df_new['TechSupport'] == 'No')
        )
    ).astype(int)

    return df_new.drop(
        ['customerID', 'Churn'],
        axis=1,
        errors='ignore'
    )

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# =====================================================================
# 3. PREPROCESAMIENTO
# EVITAR CONCATENAR TRAIN Y TEST (NO DATA LEAKAGE)
# =====================================================================
print("--- 3. Preprocesando datos ---")

# -------------------------------------------------------------
# Variables binarias
# -------------------------------------------------------------
binarias = [
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling'
]

for col in binarias:

    train_fe[col] = train_fe[col].map({
        'Yes': 1,
        'No': 0
    })

    test_fe[col] = test_fe[col].map({
        'Yes': 1,
        'No': 0
    })

# Gender
train_fe['gender'] = train_fe['gender'].map({
    'Male': 1,
    'Female': 0
})

test_fe['gender'] = test_fe['gender'].map({
    'Male': 1,
    'Female': 0
})

# -------------------------------------------------------------
# Variables categóricas
# -------------------------------------------------------------
categoricas = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaymentMethod'
]

# -------------------------------------------------------------
# ONE HOT ENCODING SEPARADO
# -------------------------------------------------------------
train_encoded = pd.get_dummies(
    train_fe,
    columns=categoricas,
    drop_first=True
)

test_encoded = pd.get_dummies(
    test_fe,
    columns=categoricas,
    drop_first=True
)

# -------------------------------------------------------------
# ALINEAR COLUMNAS
# -------------------------------------------------------------
X_train_full, X_test = train_encoded.align(
    test_encoded,
    join='left',
    axis=1,
    fill_value=0
)

# =====================================================================
# ESCALADO
# =====================================================================
print("--- Escalando datos ---")

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_full),
    columns=X_train_full.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# =====================================================================
# 4. TRAIN / VALIDATION SPLIT
# =====================================================================
print("--- 4. Separando Train y Validation ---")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled,
    y_train_raw,
    test_size=0.2,
    random_state=42,
    stratify=y_train_raw
)

# =====================================================================
# 5. MODELO 1:
# REGRESIÓN LOGÍSTICA
# =====================================================================
print("--- 5. Entrenando Logistic Regression ---")

lr_base = LogisticRegression(
    class_weight='balanced',
    max_iter=3000,
    random_state=42
)

parametros_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

busqueda_lr = GridSearchCV(
    lr_base,
    parametros_lr,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

busqueda_lr.fit(X_tr, y_tr)

mejor_lr = busqueda_lr.best_estimator_

print(f"\nMejor F1 LR: {busqueda_lr.best_score_:.4f}")
print("Mejores parámetros LR:")
print(busqueda_lr.best_params_)

# =====================================================================
# 6. MODELO 2:
# SVM (Support Vector Machine)
# =====================================================================
print("\n--- 6. Entrenando Support Vector Machine (SVM) ---")
print("Nota: El cálculo de probabilidades en SVM puede tardar unos segundos...")

# IMPORTANTE: probability=True para poder optimizar el umbral luego
svm_base = SVC(
    class_weight='balanced',
    probability=True, 
    random_state=42
)

# Parámetros básicos para no saturar la memoria
parametros_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear']
}

busqueda_svm = GridSearchCV(
    svm_base,
    parametros_svm,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

busqueda_svm.fit(X_tr, y_tr)

mejor_svm = busqueda_svm.best_estimator_

print(f"\nMejor F1 SVM: {busqueda_svm.best_score_:.4f}")
print("Mejores parámetros SVM:")
print(busqueda_svm.best_params_)

# =====================================================================
# 7. SELECCIONAR EL MEJOR MODELO
# =====================================================================
print("\n--- 7. Seleccionando Mejor Modelo ---")

if busqueda_svm.best_score_ > busqueda_lr.best_score_:
    mejor_modelo = mejor_svm
    nombre_modelo = "Support Vector Machine"
else:
    mejor_modelo = mejor_lr
    nombre_modelo = "Logistic Regression"

print(f"Modelo seleccionado: {nombre_modelo}")

# =====================================================================
# PROBABILIDADES
# =====================================================================
y_val_proba = mejor_modelo.predict_proba(X_val)[:, 1]

# =====================================================================
# 8. OPTIMIZAR UMBRAL USANDO F1-SCORE
# =====================================================================
print("\n--- 8. Buscando Umbral Óptimo con F1-Score ---")

mejor_f1 = 0
umbral_optimo = 0.5

for umbral_prueba in np.arange(0.10, 0.90, 0.01):

    y_pred_prueba = (
        y_val_proba >= umbral_prueba
    ).astype(int)

    f1 = f1_score(
        y_val,
        y_pred_prueba
    )

    if f1 > mejor_f1:
        mejor_f1 = f1
        umbral_optimo = umbral_prueba

print(f"-> Mejor umbral encontrado: {umbral_optimo:.2f}")
print(f"-> Mejor F1 encontrado: {mejor_f1:.4f}")

# =====================================================================
# 9. EVALUACIÓN FINAL
# =====================================================================
print("\n--- 9. Evaluación Final ---")

y_pred_optimo = (
    y_val_proba >= umbral_optimo
).astype(int)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred_optimo
    )
)

# =====================================================================
# MATRIZ DE CONFUSIÓN
# =====================================================================
matriz = confusion_matrix(
    y_val,
    y_pred_optimo
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    matriz,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Se Queda (0)', 'Cancela (1)'],
    yticklabels=['Se Queda (0)', 'Cancela (1)']
)

plt.title(
    f'Matriz de Confusión ({nombre_modelo})\nUmbral={umbral_optimo:.2f}',
    fontsize=14
)

plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Predicción del Modelo', fontsize=12)

plt.tight_layout()

plt.savefig(
    'matriz_confusion_optima.png',
    dpi=300
)

print("-> 'matriz_confusion_optima.png' guardada.")

# =====================================================================
# 10. IMPORTANCIA DE VARIABLES
# =====================================================================
print("\n--- 10. Generando Importancia de Variables ---")

# -------------------------------------------------------------
# Extraer pesos según el modelo ganador
# -------------------------------------------------------------
if nombre_modelo == "Logistic Regression" or (nombre_modelo == "Support Vector Machine" and mejor_modelo.kernel == 'linear'):

    coeficientes = mejor_modelo.coef_[0]

    importancia_df = pd.DataFrame({
        'Variable': X_train_full.columns,
        'Peso': coeficientes
    })

    importancia_df['Valor_Absoluto'] = (
        importancia_df['Peso'].abs()
    )

    # TOP VARIABLES
    top_10 = (
        importancia_df
        .sort_values(
            by='Valor_Absoluto',
            ascending=False
        )
        .head(10)
    )

    top_10 = top_10.sort_values(
        by='Peso',
        ascending=True
    )

    # GRÁFICO
    plt.figure(figsize=(10, 6))

    colores = [
        'red' if x > 0 else 'green'
        for x in top_10['Peso']
    ]

    plt.barh(
        top_10['Variable'],
        top_10['Peso'],
        color=colores
    )

    plt.title(
        f'Top 10 Variables Más Importantes ({nombre_modelo})',
        fontsize=14
    )

    plt.xlabel('Importancia', fontsize=12)

    plt.axvline(
        x=0,
        color='black',
        linestyle='--'
    )

    plt.tight_layout()

    plt.savefig(
        'importancia_variables.png',
        dpi=300
    )

    print("-> 'importancia_variables.png' guardada.")

else:
    print("-> Nota Teórica: El modelo ganador es un SVM con Kernel no lineal (Caja Negra).")
    print("-> No es posible graficar el peso individual de las variables de forma directa.")

# =====================================================================
# 11. PREDICCIÓN FINAL PARA KAGGLE
# =====================================================================
print("\n--- 11. Generando archivo final para Kaggle ---")

# Re-entrenar con TODO el dataset
modelo_final = mejor_modelo

modelo_final.fit(
    X_train_scaled,
    y_train_raw
)

# Probabilidades
probabilidades_kaggle = (
    modelo_final.predict_proba(X_test_scaled)[:, 1]
)

# Aplicar UMBRAL ÓPTIMO basado en F1
predicciones_kaggle = (
    probabilidades_kaggle >= umbral_optimo
).astype(int)

# Submission
submission = pd.DataFrame({
    'customerID': test_ids,
    'Churn': predicciones_kaggle
})

submission.to_csv(
    'submission_final_mejorado.csv',
    index=False
)

print("\n¡Éxito!")
print("Archivo generado:")
print("-> submission_final_mejorado.csv")

print(f"\nModelo final usado: {nombre_modelo}")
print(f"Umbral final usado: {umbral_optimo:.2f}")
print(f"F1 Validation: {mejor_f1:.4f}")