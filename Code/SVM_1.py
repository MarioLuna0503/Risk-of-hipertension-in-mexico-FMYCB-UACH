# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:07:39 2025

@author: valer
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


df_balanced = pd.read_csv("pruebas.csv")

#df_0 = df[df['riesgo_hipertension'] == 0].sample(n=1200, random_state=42)
#df_1 = df[df['riesgo_hipertension'] == 1].sample(n=1200, random_state=42)
#df_balanced = pd.concat([df_0, df_1]).sample(frac=1, random_state=42)  #unir filas

X = df_balanced.drop(columns=["riesgo_hipertension", "FOLIO_I"]) #eliminamos filas que no son de inters
y = df_balanced['riesgo_hipertension'] #definimos la fila de clases

#Dividir en train y test (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#Entrenar modelo SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

#Predecir y evaluar
y_pred = svm_model.predict(X_test_scaled)

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

# Título con todas las métricas
plt.title(f"Matriz de Confusión\nAccuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1-score: {f1:.3f}")
plt.show()

