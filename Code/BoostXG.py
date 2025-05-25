# -*- coding: utf-8 -*-
"""
Created on Sat May 24 23:08:42 2025

@author: mayit
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# 1. Cargar datos
df = pd.read_csv("pruebas.csv")

# 2. Separar características y etiquetas (ignorando la primera columna)
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Graficar datos antes de normalizar
plt.figure(figsize=(12, 6))
sns.boxplot(data=X)
plt.title("Distribución de características antes de la normalización")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. División de datos (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 4. Normalización con StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Graficar datos después de normalizar (solo entrenamiento)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train_scaled_df)
plt.title("Distribución de características después de la normalización (StandardScaler)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Convertir a DMatrix
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled, label=y_val)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# 6. Parámetros del modelo
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# 7. Entrenamiento con early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dval, 'validation')],
    early_stopping_rounds=10,
    verbose_eval=True
)

# 8. Predicciones
y_pred_prob = model.predict(dtest)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# 9. Evaluación con métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 10. Matriz de confusión con métricas en el título
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicción')
plt.ylabel('Real')

# Título con métricas
title_main = "Matriz de Confusión"
title_metrics = (
    f"Accuracy: {accuracy:.2f}  |  "
    f"Precision: {precision:.2f}  |  "
    f"Recall: {recall:.2f}  |  "
    f"F1 Score: {f1:.2f}"
)
plt.title(f"{title_main}\n{title_metrics}", fontsize=12)

plt.tight_layout()
plt.show()
