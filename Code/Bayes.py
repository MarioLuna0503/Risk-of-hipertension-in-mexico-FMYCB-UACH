# -*- coding: utf-8 -*-
"""
Created on Fri May 23 18:11:20 2025
@author: valer
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
datos = pd.read_csv("pruebas.csv")

# Se quita la primer columna (Folios)
datos = datos.iloc[:, 1:]

# Separar características y etiquetas
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values

# División: 70% entrenamiento y 30% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Normalizar con StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Función para calcular parámetros
def calcular_parametros(X, y):
    clases = np.unique(y)
    parametros = {}
    for c in clases:
        X_c = X[y == c]
        media = X_c.mean(axis=0)
        std = X_c.std(axis=0, ddof=1) + 1e-6
        prior = X_c.shape[0] / X.shape[0]
        parametros[c] = {'media': media, 'std': std, 'prior': prior}
    return parametros

def calcular_log_prob(x, media, std):
    exponent = - ((x - media) ** 2) / (2 * std ** 2)
    probabilidad = exponent - np.log(std) - 0.5 * np.log(2 * np.pi)
    return np.sum(probabilidad)

def predecir(X, parametros):
    predicciones = []
    for x in X:
        log_probs = {}
        for c in parametros:
            media = parametros[c]['media']
            std = parametros[c]['std']
            prior = np.log(parametros[c]['prior'])
            log_likelihood = calcular_log_prob(x, media, std)
            log_probs[c] = prior + log_likelihood
        predicciones.append(max(log_probs, key=log_probs.get))
    return np.array(predicciones)

def predecir_probabilidades(X, parametros):
    probabilidades = []
    for x in X:
        log_probs = {}
        for c in parametros:
            media = parametros[c]['media']
            std = parametros[c]['std']
            prior = np.log(parametros[c]['prior'])
            log_likelihood = calcular_log_prob(x, media, std)
            log_probs[c] = prior + log_likelihood
        log_prob_vals = np.array(list(log_probs.values()))
        max_log = np.max(log_prob_vals)
        probs_exp = np.exp(log_prob_vals - max_log)
        probs = probs_exp / np.sum(probs_exp)
        probabilidades.append(probs)
    return np.array(probabilidades)

# Entrenamiento
parametros = calcular_parametros(X_train, y_train)

# No hay validación, se usan directamente los parámetros aprendidos
best_parametros = parametros

# Evaluación final en test
y_pred = predecir(X_test, best_parametros)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Métricas en el conjunto de prueba ---")
print(f"Exactitud (Accuracy):      {accuracy:.4f}")
print(f"Precisión (Precision):     {precision:.4f}")
print(f"Estabilidad (Recall):      {recall:.4f}")
print(f"Desempeño (F1-Score):      {f1:.4f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Matriz de Confusión\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
plt.tight_layout()
plt.show()

# Probabilidades
y_pred_prob = predecir_probabilidades(X_test, best_parametros)
clases_ordenadas = sorted(np.unique(y))
columnas = [f'Prob de clase {c}' for c in clases_ordenadas]
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=columnas)
print("\nPrimeras 10 filas de probabilidades predichas:")
print(y_pred_prob_df.head(10))

# Histograma
y_pred1 = y_pred_prob[:, clases_ordenadas.index(1)]
plt.rcParams['font.size'] = 12
plt.hist(y_pred1, bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma de probabilidades predichas para clase 1')
plt.xlabel('Probabilidades predichas')
plt.ylabel('Frecuencia')
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# Funciones para uso futuro
def entrenar_y_preparar():
    global best_parametros, scaler
    return best_parametros, scaler

def predecir_una_muestra(muestra, parametros, scaler):
    muestra_scaled = scaler.transform([muestra])
    proba = predecir_probabilidades(muestra_scaled, parametros)
    pred = predecir(muestra_scaled, parametros)
    return int(pred[0]), proba[0]
