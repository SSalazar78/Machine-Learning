# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:59:35 2026

@author: Susana A.S.R
"""
#Importación de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score, accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time

#Carga de datos
ruta='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
columnas = [
    'status_checking', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings', 'employment_duration', 'installment_rate',
    'personal_status', 'other_debtors', 'residence_duration', 'property',
    'age', 'other_installment_plans', 'housing', 'existing_credits',
    'job', 'num_people_liable', 'telephone', 'foreign_worker', 'target'
]

df = pd.read_csv(ruta, sep=' ', header= None, names = columnas)

#Descripción del Dataset 
# 1)Infomación general
print(f"Total de registros (filas): {df.shape[0]}")
print(f"Total de variables (columnas): {df.shape[1]}")

print("\nTipos de datos")
print(df.dtypes)

#Análisis de variable objetivo (target)
print("Información del target:")
print(f"\nConteo absoluto:{df['target'].value_counts()}")
print(f"\nPorcentajes:{df['target'].value_counts(normalize=True) * 100}")

# 2) Estadistica descriptiva de las varibles numéricas

print("Descripción del dataset")
variables_numericas = ['duration', 'credit_amount', 'installment_rate', 
                       'residence_duration', 'age', 'existing_credits', 
                       'num_people_liable']

tabla1 = [] #Se crea una lista vacia para la presentación de las estadisticas por variable
for var in variables_numericas:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    outliers = df[(df[var] < limite_inf) | (df[var] > limite_sup)]
    
    tabla1.append({
        'Media': df[var].mean(),
        'Mediana': df[var].median(),
        'Desv. Est.': df[var].std(),
        'Mín': df[var].min(),
        'Máx': df[var].max(),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Outliers (n)': len(outliers),
        'Outliers (%)': (len(outliers) / len(df)) * 100
    })
    
# Tabla con variables como columnas y estadísticas como filas
df_stats = pd.DataFrame(tabla1, index=variables_numericas).T

pd.set_option('display.float_format', '{:.2f}'.format)
print("\n", df_stats)

#Historigramas de las variables numéricas
# Creamos una cuadrícula de 4 filas x 2 columnas
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
axes = axes.flatten() # Aplanamos el array para iterar fácilmente


for i, var in enumerate(variables_numericas):
    axes[i].hist(df[var], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribución de {var.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Frecuencia')
    axes[i].set_xlabel(var.replace("_", " ").title())
    axes[i].grid(axis='y', alpha=0.3)
    
# Ocultamos el último subplot vacío
axes[-1].axis('off')

plt.suptitle('Distribución de Variables Numéricas', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('grid_histogramas_numericas.png', dpi=300, bbox_inches='tight')
plt.show()

#Matriz de correlación entre el target y las variables numéricas
var_corr = variables_numericas + ['target']

matriz_corr = df[var_corr].corr()

print("\n Matriz de correlación: variables numéricas")
print(matriz_corr)

#Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Matriz de Correlación - German Credit Data', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300)
plt.show()


#Análisis de las variables categóricas
variables_categoricas = ['status_checking', 'credit_history', 'purpose', 
                         'savings', 'employment_duration', 'housing', 'job',
                         'personal_status', 'other_debtors', 'property',
                         'other_installment_plans', 'telephone', 'foreign_worker']


for var in variables_categoricas:
    conteos = df[var].value_counts()
    porcentajes = df[var].value_counts(normalize=True) * 100
    
    print(f"\n{var.upper()}")
    print(f"  Número de categorías: {df[var].nunique()}")
    for categoria in conteos.index:
        print(f"    {categoria}: {conteos[categoria]} ({porcentajes[categoria]:.1f}%)")

#Visualización de las variables categóricas
#Creamos otra cuadrícula de 3 filas x 5 columnas
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(14, 28))
axes = axes.flatten()

for i, var in enumerate(variables_categoricas):
    conteos = df[var].value_counts()
    # Generar colores tipo 'viridis' manualmente
    colors = plt.cm.viridis(np.linspace(0, 1, len(conteos)))
    
    axes[i].bar(range(len(conteos)), conteos.values, color=colors, edgecolor='black')
    axes[i].set_title(f'{var.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Frecuencia')
    
    # Configurar etiquetas del eje X
    axes[i].set_xticks(range(len(conteos)))
    axes[i].set_xticklabels(conteos.index, rotation=45, ha='right', fontsize=9)
    axes[i].grid(axis='y', alpha=0.3)

# Ocultamos los subplot vacios
axes[13].axis('off')
axes[14].axis('off')

plt.suptitle('Distribución de Variables Categóricas', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('grid_barras_categoricas.png', dpi=300, bbox_inches='tight')
plt.show()

#3) Desarrollo
#3.1) Reduccion dimensional

print("Reducción dimensional")
#Para poder aplicar reducción dimensional, primero hay que convertir las variables categóricas en numericas, esto puede hacerse con One-Hot Encoder

#Separar target del dataset y nombrarlo X para aplicar One-Hot Encoder
X = df.drop('target', axis=1)
y = df['target']
X_encoded = pd.get_dummies(X, drop_first=True) #One-Hot Enconder

#Estandarizar los datos para la reducción dimensional
scaler = StandardScaler() #Lo vuelvo a usar en (3.4)Primer modelo)
X_scaled = scaler.fit_transform(X_encoded)

#Aplicar PCA para ver cuánta varianza explica cada componente
pca_completo = PCA()
pca_completo.fit(X_scaled)

# Calcular varianza explicada y acumulada
varianza_explicada = pca_completo.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

# Encontrar cuántos componentes son necesarios para conservar el 90% y 95% de la información
n_90 = np.argmax(varianza_acumulada >= 0.90) + 1
n_95 = np.argmax(varianza_acumulada >= 0.95) + 1


print("RESULTADOS DEL ANÁLISIS PCA:")
print(f"\n Variables originales: {X_encoded.shape[1]}")
print(f"\n Para retener 90% de información: {n_90} componentes")
print(f"\n Para retener 95% de información: {n_95} componentes")


# Gráfica de varianza acumulada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o', linestyle='--', color='b', linewidth=1, label='Varianza Acumulada')
plt.axhline(y=0.90, color='red', linestyle=':', linewidth=1, label='Umbral 90%')
plt.axhline(y=0.95, color='green', linestyle=':', linewidth=1, label='Umbral 95%')
plt.scatter([n_90], [varianza_acumulada[n_90-1]], color='red', s=150, zorder=5)
plt.scatter([n_95], [varianza_acumulada[n_95-1]], color='green', s=150, zorder=5)
plt.title('Análisis de Varianza Explicada - PCA', fontsize=14, fontweight='bold')
plt.xlabel('Número de Componentes', fontsize=12)
plt.ylabel('Varianza Acumulada', fontsize=12)
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig('pca_varianza.png', dpi=300)
plt.show()

#Determinar las variables de peso para el 95% de la información
pca_cargas = PCA(n_components=n_95)
pca_cargas.fit(X_scaled)

#Dataframe de las componentes
cargas = pd.DataFrame(pca_cargas.components_, columns=X_encoded.columns, index= [f'Componente_{i+1}' for i in range(pca_cargas.n_components_)] )

#Transponer para que las variables queden en las filas
print("\nCargas factoriales completas:")
print(cargas.T.round(4))

# Crear Heatmap
plt.figure(figsize=(16, 20))

sns.heatmap( cargas.T, cmap='RdBu_r', center=0, annot=False, fmt='.2f',cbar_kws={'label': 'Carga Factorial'}, linewidths=0.5,linecolor='gray')
plt.title('Cargas Factoriales PCA',fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Componentes Principales', fontsize=12, labelpad=10)
plt.ylabel('Variables Originales', fontsize=12, labelpad=10)
plt.xticks(rotation=90, fontsize=9)  # ← Cambié de 0 a 90
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('heatmap_cargas_factoriales.png', dpi=300, bbox_inches='tight')
plt.show()


#3.2) Comparativo entre modelos no supervisados.
print("Comportamiento de los modelos no supervisados")
#Medir el tiempo de entrenamiento para el punto (3.6)
start_time = time.time() 

#Primer modelo de entrenamiento: K-MEANS
kmeans = KMeans(n_clusters=2, random_state=42) 
kmeans_label = kmeans.fit_predict(X_scaled)

train_timekm = time.time() - start_time

#Evaluando el modelo
silhouette_km = silhouette_score(X_scaled, kmeans_label)
davies_km = davies_bouldin_score(X_scaled, kmeans_label)

print("Parámetros: K-Means")
print(f"Tiempo de entrenamiento: {train_timekm:.4f} segundos")
print(f"Silhouette Score: {silhouette_km:.4f}")
print(f"Davies-Bouldin Index: {davies_km:.4f}")

#Segundo modelo de entrenamiento: Agglomerative clustering
start_time = time.time()

Agg = AgglomerativeClustering(n_clusters=2)
Agg_label = Agg.fit_predict(X_scaled)

train_timeAgg = time.time() - start_time

# Evaluar el modelo
silhouette_Agg = silhouette_score(X_scaled, Agg_label)
davies_Agg = davies_bouldin_score(X_scaled, Agg_label)

print("Parámetros: Agglomerative clustering")
print(f"Tiempo de entrenamiento: {train_timeAgg:.4f} segundos")
print(f"Silhouette Score: {silhouette_Agg:.4f}")
print(f"Davies-Bouldin Index: {davies_Agg:.4f}")

#Tabla comparativa
comparativa = pd.DataFrame({
    'Modelo': ['K-Means', 'Agglomerative'],
    'Silhouette Score': [silhouette_km, silhouette_Agg],
    'Davies-Bouldin': [davies_km, davies_Agg],
    'Tiempo (s)': [train_timekm, train_timeAgg]
})
print(comparativa.to_string(index=False))

#3.3)Describir qué conjunto de características comparten los resultados obtenidos con los modelos NO supervisados
print("Descripción de los clusters.")
print("Análisis de clusters - KMEANS")
# Crear DataFrame con etiquetas de cluster
df_kmeans = X.copy()
df_kmeans['cluster'] = kmeans_label

#Resumen promedio de cada variable dentro de cada cluster
perfil_kmeans = (df_kmeans.groupby('cluster')[variables_numericas].mean()) #Perfil de variables
print("\nPromedio de variables numéricas por cluster:")
print(perfil_kmeans.round(2))

#Diferencias de las variables entre clusters
diferencias_kmeans = abs(perfil_kmeans.loc[0] - perfil_kmeans.loc[1]) 
diferencias_kmeans = diferencias_kmeans.sort_values(ascending=False) #Solo para ordenar

print("\nVariables con mayor diferencia entre clusters (K-Means):")
print(diferencias_kmeans)

#Distribución del target por cluster porcentaje
tabla_target_kmeans = (pd.crosstab(kmeans_label, y, normalize='index') * 100)
print("\nDistribución del target por cluster (%):")
print(tabla_target_kmeans.round(2))
 
#Variables categóricas por cluster porcentaje
for var in variables_categoricas:
    print(f"\n{var.upper()}")
    tabla = (pd.crosstab(df_kmeans['cluster'],df[var],normalize='index') * 100)
    print(tabla.round(2))
    
print("Análisis de clusters - AGGLOMERATIVE")
#Crear dataframe con etiquetas de cluster
df_agg = X.copy()
df_agg['cluster'] = Agg_label

#Resumen promedio de cada variable dentro de cada cluster
perfil_agg = (df_agg.groupby('cluster')[variables_numericas].mean())
print("\nPromedio de variables numéricas por cluster:")
print(perfil_agg.round(2))

#Diferencia de las variables entre clusters
diferencias_agg = abs(perfil_agg.loc[0] - perfil_agg.loc[1])
diferencias_agg = diferencias_agg.sort_values(ascending=False) #Solo para ordenar resultados

print("\nVariables con mayor diferencia entre clusters (Agglomerative):")
print(diferencias_agg)

#Distribución del target por cluster en porcentaje
tabla_target_agg = (pd.crosstab(Agg_label,y, normalize='index') * 100)
print("\nDistribución del target por cluster (%):")
print(tabla_target_agg.round(2))

#Variables categoricas por cluster en porcentaje
for var in variables_categoricas:
    print(f"\n{var.upper()}")
    tabla = (pd.crosstab(df_agg['cluster'],df[var],normalize='index') * 100)
    print(tabla.round(2))
    
#3.4) Modelos de aprendizaje supervisados

print("Modelos supervisados")
#Separación de los datos para el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

#Primer modelo: Logistic Regression
#Escalar los datos
X_train_scal = scaler.fit_transform(X_train)
X_test_scal = scaler.transform(X_test)

start_time = time.time()

#Creación y entrenamiento
lr = LogisticRegression(max_iter= 1000, random_state= 42, C=1.0)

lr.fit(X_train_scal, y_train)

train_time_lr = time.time() - start_time

#Predicciones del modelo
y_train_lr = lr.predict(X_train_scal)
y_test_lr = lr.predict(X_test_scal)

# Métricas en prueba
acc_lr = accuracy_score(y_test, y_test_lr)
prec_lr = precision_score(y_test, y_test_lr, average='weighted')
rec_lr = recall_score(y_test, y_test_lr, average='weighted')
f1_lr = f1_score(y_test, y_test_lr, average='weighted')


# Reporte de clasificación detallado
print("\nREPORTE DE CLASIFICACIÓN")
print(classification_report(y_test, y_test_lr, target_names=['Buen Crédito (1)', 'Mal Crédito (2)']))


#Segundo modelo: Random Forest Classifier
start_time = time.time()

rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs= -1)
rf.fit(X_train, y_train)

train_time_rf = time.time() - start_time

# Predicciones
y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(X_test)


# Métricas en prueba
acc_rf = accuracy_score(y_test, y_test_rf)
prec_rf = precision_score(y_test, y_test_rf, average='weighted')
rec_rf = recall_score(y_test, y_test_rf, average='weighted')
f1_rf = f1_score(y_test, y_test_rf, average='weighted')

# Reporte de clasificación detallado
print("\nREPORTE DE CLASIFICACIÓN")
print(classification_report(y_test, y_test_rf, target_names=['Buen Crédito (1)', 'Mal Crédito (2)']))


#Tabla comparativa
print("Tabla compartiva de métricas en prueba")
comparativa_supervisados = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Tiempo (s)'],
    'Random Forest': [f"{acc_rf*100:.2f}%", f"{prec_rf:.4f}", f"{rec_rf:.4f}", f"{f1_rf:.4f}", f"{train_time_rf:.4f}"],
    'Logistic Regression': [f"{acc_lr*100:.2f}%", f"{prec_lr:.4f}", f"{rec_lr:.4f}", f"{f1_lr:.4f}", f"{train_time_lr:.4f}"]
})
print(comparativa_supervisados.to_string(index=False))


#Matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#Random Forest
cm_rf = confusion_matrix(y_test, y_test_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Random Forest\nAccuracy: {acc_rf*100:.2f}%', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Real')
axes[0].set_xticklabels(['Bueno (1)', 'Malo (2)'])
axes[0].set_yticklabels(['Bueno (1)', 'Malo (2)'])

#Logistic Regression
cm_lr = confusion_matrix(y_test, y_test_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title(f'Logistic Regression\nAccuracy: {acc_lr*100:.2f}%', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicción')
axes[1].set_ylabel('Real')
axes[1].set_xticklabels(['Bueno (1)', 'Malo (2)'])
axes[1].set_yticklabels(['Bueno (1)', 'Malo (2)'])

plt.suptitle('Matrices de Confusión: Modelos Supervisados', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('matrices_confusion_supervisados.png', dpi=300)
plt.show()





