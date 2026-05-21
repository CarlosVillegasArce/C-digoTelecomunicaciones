import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Semilla global para replicabilidad
SEED = 42

# =====================================================================
# 1. CARGA DE DATOS Y LIMPIEZA BÁSICA
# =====================================================================
print("--- 1. Cargando datos ---")

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

y_train_raw = train['Churn'].map({'Yes': 1, 'No': 0})

test_ids = test['customerID']

print(f"Train shape: {train.shape} | Test shape: {test.shape}")
print(f"Distribución Churn -> No: {(y_train_raw==0).sum()} | Yes: {(y_train_raw==1).sum()}")

# =====================================================================
# 2. INGENIERÍA DE VARIABLES
# =====================================================================
print("\n--- 2. Creando nuevas variables ---")

def engineer_features(df):

    df_new = df.copy()

    # Forzar variables numéricas
    for col in ['TotalCharges', 'tenure', 'MonthlyCharges']:
        df_new[col] = pd.to_numeric(
            df_new[col], errors='coerce'
        ).fillna(0)

    # Variables de servicios
    servicios = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    df_new['num_servicios'] = df_new[servicios].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )

    # Variables financieras
    df_new['gasto_promedio_real'] = (
        df_new['TotalCharges'] / df_new['tenure'].clip(lower=1)
    )

    df_new['costo_por_servicio'] = (
        df_new['MonthlyCharges'] / df_new['num_servicios'].clip(lower=1)
    )

    # Variables de comportamiento
    df_new['cliente_nuevo']    = (df_new['tenure'] <= 3).astype(int)
    df_new['contrato_mensual'] = (df_new['Contract'] == 'Month-to-month').astype(int)
    df_new['fibra_sin_soporte'] = (
        (df_new['InternetService'] == 'Fiber optic') &
        (df_new['TechSupport'] == 'No')
    ).astype(int)

    return df_new.drop(['customerID', 'Churn'], axis=1, errors='ignore')


train_fe = engineer_features(train)
test_fe  = engineer_features(test)

# =====================================================================
# 3. PREPROCESAMIENTO (SIN DATA LEAKAGE)
# =====================================================================
print("--- 3. Preprocesando datos ---")

binarias = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binarias:
    train_fe[col] = train_fe[col].map({'Yes': 1, 'No': 0})
    test_fe[col]  = test_fe[col].map({'Yes': 1, 'No': 0})

train_fe['gender'] = train_fe['gender'].map({'Male': 1, 'Female': 0})
test_fe['gender']  = test_fe['gender'].map({'Male': 1, 'Female': 0})

categoricas = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]

train_encoded = pd.get_dummies(train_fe, columns=categoricas, drop_first=True)
test_encoded  = pd.get_dummies(test_fe,  columns=categoricas, drop_first=True)

# Alinear columnas (evita desajuste por categorías ausentes en test)
X_train_full, X_test = train_encoded.align(
    test_encoded, join='left', axis=1, fill_value=0
)

# =====================================================================
# ESCALADO
# =====================================================================
print("--- Escalando datos ---")

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_full), columns=X_train_full.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns
)

# =====================================================================
# 4. TRAIN / VALIDATION SPLIT (estratificado)
# =====================================================================
print("--- 4. Separando Train y Validation (80/20, estratificado) ---")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train_raw,
    test_size=0.2, random_state=SEED, stratify=y_train_raw
)

# =====================================================================
# 5. MODELO 1: REGRESIÓN LOGÍSTICA
# =====================================================================
print("\n--- 5. Entrenando Regresión Logística (GridSearchCV, cv=5) ---")

lr_base = LogisticRegression(
    class_weight='balanced', max_iter=3000, random_state=SEED
)

parametros_lr = {
    'C':       [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver':  ['lbfgs', 'liblinear']
}

busqueda_lr = GridSearchCV(
    lr_base, parametros_lr, cv=5, scoring='f1', n_jobs=-1
)
busqueda_lr.fit(X_tr, y_tr)
mejor_lr = busqueda_lr.best_estimator_

print(f"  Mejor F1 CV (LR) : {busqueda_lr.best_score_:.4f}")
print(f"  Mejores params LR: {busqueda_lr.best_params_}")

# =====================================================================
# 6. MODELO 2: SUPPORT VECTOR MACHINE
# =====================================================================
print("\n--- 6. Entrenando SVM (GridSearchCV, cv=3) ---")

svm_base = SVC(
    class_weight='balanced', probability=True, random_state=SEED
)

parametros_svm = {
    'C':      [0.1, 1, 10],
    'kernel': ['rbf', 'linear']
}

busqueda_svm = GridSearchCV(
    svm_base, parametros_svm, cv=3, scoring='f1', n_jobs=-1
)
busqueda_svm.fit(X_tr, y_tr)
mejor_svm = busqueda_svm.best_estimator_

print(f"  Mejor F1 CV (SVM) : {busqueda_svm.best_score_:.4f}")
print(f"  Mejores params SVM: {busqueda_svm.best_params_}")

# =====================================================================
# TABLA COMPARATIVA DE MODELOS (umbral = 0.5)
# =====================================================================
print("\n--- Comparativa con umbral = 0.5 ---")

for nombre, modelo in [("Logistic Regression", mejor_lr), ("SVM", mejor_svm)]:
    proba = modelo.predict_proba(X_val)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    print(f"  {nombre}: accuracy={accuracy_score(y_val, pred):.4f}  f1={f1_score(y_val, pred):.4f}")

# =====================================================================
# 7. SELECCIÓN DEL MEJOR MODELO (por F1 en CV)
# =====================================================================
print("\n--- 7. Seleccionando Mejor Modelo ---")

if busqueda_svm.best_score_ > busqueda_lr.best_score_:
    mejor_modelo  = mejor_svm
    nombre_modelo = "Support Vector Machine"
else:
    mejor_modelo  = mejor_lr
    nombre_modelo = "Logistic Regression"

print(f"  Modelo seleccionado: {nombre_modelo}")

# =====================================================================
# PROBABILIDADES DEL MODELO GANADOR
# =====================================================================
y_val_proba = mejor_modelo.predict_proba(X_val)[:, 1]

# =====================================================================
# 8. OPTIMIZAR UMBRAL CON F1-SCORE
# =====================================================================
print("\n--- 8. Buscando Umbral Óptimo ---")

mejor_f1     = 0
umbral_optimo = 0.5

for umbral_prueba in np.arange(0.10, 0.90, 0.01):
    y_pred_prueba = (y_val_proba >= umbral_prueba).astype(int)
    f1 = f1_score(y_val, y_pred_prueba)
    if f1 > mejor_f1:
        mejor_f1     = f1
        umbral_optimo = umbral_prueba

print(f"  Umbral óptimo : {umbral_optimo:.2f}")
print(f"  Mejor F1 val  : {mejor_f1:.4f}")

# =====================================================================
# 9. EVALUACIÓN FINAL
# =====================================================================
print("\n--- 9. Evaluación Final ---")

y_pred_optimo = (y_val_proba >= umbral_optimo).astype(int)
acc_final     = accuracy_score(y_val, y_pred_optimo)

print(f"  Accuracy (umbral óptimo): {acc_final:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_optimo))

# =====================================================================
# MATRIZ DE CONFUSIÓN
# =====================================================================
matriz = confusion_matrix(y_val, y_pred_optimo)
TN, FP, FN, TP = matriz.ravel()

print(f"  TN={TN}  FP={FP}  FN={FN}  TP={TP}")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    matriz, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Se Queda (0)', 'Cancela (1)'],
    yticklabels=['Se Queda (0)', 'Cancela (1)'],
    ax=ax, annot_kws={"size": 13}
)
ax.set_title(
    f'Matriz de Confusión — {nombre_modelo}\nUmbral óptimo = {umbral_optimo:.2f}',
    fontsize=13
)
ax.set_ylabel('Valor Real', fontsize=11)
ax.set_xlabel('Predicción del Modelo', fontsize=11)
plt.tight_layout()
plt.savefig('matriz_confusion_optima.png', dpi=200)
plt.close()
print("  -> 'matriz_confusion_optima.png' guardada.")

# =====================================================================
# 10. ANÁLISIS DE COSTO-BENEFICIO
# =====================================================================
print("\n--- 10. Análisis de Costo-Beneficio ---")

COSTO_FP = 10    # Costo de contactar cliente que no iba a cancelar
COSTO_FN = 100   # Costo de perder un cliente que sí iba a cancelar

costo_total = FP * COSTO_FP + FN * COSTO_FN
print(f"  Falsos Positivos (FP): {FP}  ->  costo = ${FP * COSTO_FP:,}")
print(f"  Falsos Negativos (FN): {FN}  ->  costo = ${FN * COSTO_FN:,}")
print(f"  Costo Total Esperado : ${costo_total:,}")

# =====================================================================
# 11. IMPORTANCIA DE VARIABLES
# =====================================================================
print("\n--- 11. Importancia de Variables ---")

if nombre_modelo == "Logistic Regression" or \
   (nombre_modelo == "Support Vector Machine" and mejor_modelo.kernel == 'linear'):

    coeficientes = mejor_modelo.coef_[0]
    importancia_df = pd.DataFrame({
        'Variable': X_train_full.columns,
        'Peso': coeficientes
    })
    importancia_df['Valor_Absoluto'] = importancia_df['Peso'].abs()

    top_10 = importancia_df.sort_values('Valor_Absoluto', ascending=False).head(10)
    top_10 = top_10.sort_values('Peso', ascending=True)

    colores = ['#d62728' if x > 0 else '#2ca02c' for x in top_10['Peso']]

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.barh(top_10['Variable'], top_10['Peso'], color=colores)
    ax2.set_title(
        f'Top 10 Variables Más Importantes — {nombre_modelo}',
        fontsize=13
    )
    ax2.set_xlabel('Coeficiente (impacto en probabilidad de churn)', fontsize=11)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('importancia_variables.png', dpi=200)
    plt.close()
    print("  -> 'importancia_variables.png' guardada.")

else:
    print("  -> SVM con kernel no lineal: importancia de variables no disponible directamente.")

# =====================================================================
# 12. PREDICCIÓN FINAL PARA KAGGLE
# =====================================================================
print("\n--- 12. Generando archivo final para Kaggle ---")

modelo_final = mejor_modelo
modelo_final.fit(X_train_scaled, y_train_raw)

probabilidades_kaggle = modelo_final.predict_proba(X_test_scaled)[:, 1]
predicciones_kaggle   = (probabilidades_kaggle >= umbral_optimo).astype(int)

submission = pd.DataFrame({
    'customerID': test_ids,
    'Churn': predicciones_kaggle
})
submission.to_csv('submission_final_mejorado.csv', index=False)

# =====================================================================
# RESUMEN FINAL
# =====================================================================
print("\n" + "="*55)
print("RESUMEN FINAL")
print("="*55)
print(f"  Modelo ganador       : {nombre_modelo}")
print(f"  Mejor params         : {busqueda_lr.best_params_ if nombre_modelo == 'Logistic Regression' else busqueda_svm.best_params_}")
print(f"  F1 CV (train)        : {busqueda_lr.best_score_:.4f}" if nombre_modelo == 'Logistic Regression' else f"  F1 CV (train)        : {busqueda_svm.best_score_:.4f}")
print(f"  Umbral óptimo        : {umbral_optimo:.2f}")
print(f"  F1 validación        : {mejor_f1:.4f}")
print(f"  Accuracy validación  : {acc_final:.4f}")
print(f"  TN={TN}  FP={FP}  FN={FN}  TP={TP}")
print(f"  Costo Total Esperado : ${costo_total:,}")
print(f"  Submission           : submission_final_mejorado.csv")
print("="*55)
