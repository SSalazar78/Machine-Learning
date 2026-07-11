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
variables_numericas = ['duration', 'credit_amount', 'installment_rate','residence_duration', 'age', 
                       'existing_credits','num_people_liable']

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

###################################################################
#Hay que moverlo despues del OHE
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
###################################################################

#Análisis de las variables categóricas
variables_categoricas = ['status_checking', 'credit_history', 'purpose','savings', 'employment_duration', 'housing', 'job', 
                         'personal_status', 'other_debtors', 'property','other_installment_plans', 'telephone', 'foreign_worker']

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

#Modificación: Digramas de dispersión 
# 1) Duración vs monto
sns.scatterplot(data = df, x = "duration", y = "credit_amount", hue = "target", alpha=0.5,
    palette=["green", "red"])
plt.title("Duración vs Monto del Crédito por Riesgo")
plt.show()

# 2) Edad vs monto
sns.scatterplot(data = df, x = "age", y = "credit_amount", hue = "target", alpha=0.5,
    palette=["green", "red"])
plt.title("Edad del Cliente vs Monto Solicitado")
plt.show()

# 3) Monto vs propósito
sns.scatterplot(data = df, x = "age", y = "duration", hue = "target", alpha=0.5,
    palette=["green", "red"])
plt.title("Edad vs Duración del Crédito")
plt.show()

# 4) Monto vs propósito.
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df, x="purpose", y="credit_amount", hue="target", alpha=0.7, 
              palette = ["green", "red"], size=2)
plt.xticks(rotation=45)
plt.title("Monto del Crédito según Propósito")
plt.show()

# 5) Edad vs historial
sns.stripplot(data=df, x="credit_history", y="age", hue="target", alpha=0.5,
              palette=["green", "red"])
plt.title("Edad según Historial Crediticio y Riesgo")
plt.show()

# 6) Monto vs tipo de vivienda
sns.stripplot(data=df, x="housing", y="credit_amount", hue="target", alpha=0.5,
              palette=["green", "red"])
plt.title("Monto del Crédito según Tipo de Vivienda")
plt.show()


#3) Desarrollo
#3.1) Reduccion dimensional
print("Reducción dimensional")

#Crear variables a partir de variables del dataset.
#Reduccion de categorias en variables categoricas puntuales.

personal_mapping = {
    'A91': 'male',
    'A92': 'female',
    'A93': 'male',
    'A94': 'male',
    'A95': 'female'
    } 
df['personal_grouped'] = df['personal_status'].map(personal_mapping)

job_mapping = {
    'A171': 'unskilled',
    'A172': 'unskilled',
    'A173': 'skilled',
    'A174': 'skilled'
    }
df['job_grouped'] = df['job'].map(job_mapping)

purpose_mapping ={
    'A40': 'vehicle',  
    'A41': 'vehicle',  
    'A42': 'household',
    'A43': 'household',
    'A44': 'household',
    'A45': 'household',
    'A46': 'education',
    'A47': 'others',  
    'A48': 'education',
    'A49': 'business',
    'A410': 'others'
    }
df['purpose_grouped'] = df['purpose'].map(purpose_mapping)

employment_duration_mapping = {
    'A71': 'unemployed',
    'A72': 'short_term',  
    'A73': 'short_term',  
    'A74': 'long_term',   
    'A75': 'long_term'    
    }
df['employment_grouped'] = df['employment_duration'].map(employment_duration_mapping)

#Retirar las anteriores
df = df.drop(columns = ['personal_status', 'job', 'purpose', 'employment_duration'])

#Creacion de variables numericas que aportan informacion 
df['credit_weight'] = df['existing_credits'] * df['installment_rate']

df['residence_ratio'] = df['residence_duration']/df['age']

#Para poder aplicar reducción dimensional, primero hay que convertir las variables categóricas en 
#numericas, esto puede hacerse con One-Hot Encoder
#Separar target del dataset y nombrarlo X 
X = df.drop('target', axis=1)
y = df['target']
X_encoded = pd.get_dummies(X, drop_first=True) #One-Hot Enconder

#Ver las correlaciones con los dummies y las variables nuevas
corr_target = X_encoded.corrwith(y).abs().sort_values(ascending=False)

#Mostrar el Top 20
print("\n TOP 20 VARIABLES MÁS CORRELACIONADAS CON EL TARGET")
print(corr_target.head(20))

#Gráfico
plt.figure(figsize=(10, 12))
corr_target.head(20).plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Top 20 Variables con Mayor Correlación con Target', fontsize=14, fontweight='bold')
plt.xlabel('Correlación (valor absoluto)', fontsize=12)
plt.gca().invert_yaxis() 
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

#Dividir los datos en pruebas y testeo
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Aplicando PCA para la reduccion
pca = PCA(n_components=0.90, random_state=42)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Variables originales: {X_train_scaled.shape[1]}")
print(f"Variables después de PCA: {X_train_pca.shape[1]}")


#3.2) Comparativo entre modelos no supervisados.
print("Comportamiento de los modelos no supervisados")
#Inicio del entrenamiento
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_label = kmeans.fit_predict(X_train_pca)

print("Tamaño de cada cluster:")
for i in range(2):
    print(f"\nCluster {i}: {sum(kmeans_label == i)} clientes")

# Crear DataFrame con cluster y target
df_clusters = pd.DataFrame({'Cluster': kmeans_label,'Target': y_train.values})

cross_tab = pd.crosstab(df_clusters['Cluster'], df_clusters['Target'], margins=True)
print(cross_tab)

#Calcular las métricas 
silhouette_km = silhouette_score(X_train_pca, kmeans_label)
davies_km = davies_bouldin_score(X_train_pca, kmeans_label)

print("Parámetros: K-Means")
print(f"Silhouette Score: {silhouette_km:.4f}")
print(f"Davies-Bouldin Index: {davies_km:.4f}")

#Segundo modelo de clustering
Agg = AgglomerativeClustering(n_clusters=2)
Agg_label = Agg.fit_predict(X_train_pca)

print("Tamaño de cada cluster:")
for i in range(2):
    print(f"\nCluster {i}: {sum(Agg_label == i)} clientes")

# Evaluar el modelo
silhouette_Agg = silhouette_score(X_train_pca, Agg_label)
davies_Agg = davies_bouldin_score(X_train_pca, Agg_label)

print("Parámetros: Agglomerative clustering")
print(f"Silhouette Score: {silhouette_Agg:.4f}")
print(f"Davies-Bouldin Index: {davies_Agg:.4f}")

#Tabla comparativa
comparativa = pd.DataFrame({
    'Modelo': ['K-Means', 'Agglomerative'],
    'Silhouette Score': [silhouette_km, silhouette_Agg],
    'Davies-Bouldin': [davies_km, davies_Agg]})
print(comparativa.to_string(index=False))


#3.3)Describir qué conjunto de características comparten los resultados obtenidos con los modelos NO supervisados

#Para la comparación y saber las variables representativas se crea un DataFrame 
#base con los datos escalados (para interpretar variables originales)
#y se agregan las etiquetas de ambos modelos y el target real.
df_analisis = pd.DataFrame(X_train_scaled, columns=X_encoded.columns)
df_analisis['Cluster_KM'] = kmeans.labels_
df_analisis['Cluster_Agg'] = Agg.labels_
df_analisis['Target'] = y_train.values

#Comparación
print("Comparación según el target")
print("\nDistribución del Target por Cluster: K-Means")
tabla_km = pd.crosstab(df_analisis['Cluster_KM'], df_analisis['Target'], normalize='index') * 100
print(tabla_km.round(2))

print("\nDistribución del Target por Cluster: Agglomerative Clustering")
tabla_agg = pd.crosstab(df_analisis['Cluster_Agg'], df_analisis['Target'], normalize='index') * 100
print(tabla_agg.round(2))

#Variebles representativas

print("\n Variables representativas de los clusters por algoritmo")
#K-Means
perfil_km = df_analisis.groupby('Cluster_KM').mean()
diff_km = (perfil_km.max() - perfil_km.min()).sort_values(ascending=False)
print("\nCinco variables que definen los clusters de K-Means:")
print(diff_km.head(5))

#Agglomerative
perfil_agg = df_analisis.groupby('Cluster_Agg').mean()
diff_agg = (perfil_agg.max() - perfil_agg.min()).sort_values(ascending=False)
print("\nCinco variables que definen los clusters Aglomerative:")
print(diff_agg.head(5))
 

#3.4) Modelos de aprendizaje supervisados
print("Modelos supervisados")
#Creación y entrenamiento
lr = LogisticRegression(max_iter= 1000, random_state= 42, C=1.0)
lr.fit(X_train_pca, y_train)

#Predicciones del modelo
y_test_lr = lr.predict(X_test_pca)

# Métricas en prueba
acc_lr = accuracy_score(y_test, y_test_lr)
prec_lr = precision_score(y_test, y_test_lr, average='weighted')
rec_lr = recall_score(y_test, y_test_lr, average='weighted')
f1_lr = f1_score(y_test, y_test_lr, average='weighted')


#Segundo modelo: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs= -1)
rf.fit(X_train_pca, y_train)

# Predicciones
y_test_rf = rf.predict(X_test_pca)

# Métricas en prueba
acc_rf = accuracy_score(y_test, y_test_rf)
prec_rf = precision_score(y_test, y_test_rf, average='weighted')
rec_rf = recall_score(y_test, y_test_rf, average='weighted')
f1_rf = f1_score(y_test, y_test_rf, average='weighted')


#Tabla comparativa
print("Tabla compartiva de métricas en prueba")
comparativa_supervisados = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Random Forest': [f"{acc_rf*100:.2f}%", f"{prec_rf:.4f}", f"{rec_rf:.4f}", f"{f1_rf:.4f}"],
    'Logistic Regression': [f"{acc_lr*100:.2f}%", f"{prec_lr:.4f}", f"{rec_lr:.4f}", f"{f1_lr:.4f}"]})
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


#VARIABLES
print("Perfil de clientes")
#Regresión logistica
#Extraemos los coeficientes y los multiplicamos por los componentes del PCA
coef_modelo_lr = lr.coef_[0]
importancia_lr = np.dot(coef_modelo_lr, pca.components_)

# Creamos el dataframe con las variables originales
coeficientes = pd.DataFrame({
    'Variable': X_encoded.columns,
    'Coeficiente': importancia_lr}).sort_values('Coeficiente', key=abs, ascending=False)

#Ordenadas sin importar si su efecto es positivo o negativo
print("\n 15 VARIABLES MÁS IMPORTANTES (Regresión Logística)")
print(coeficientes.head(15).round(2))


#Random Forest 
#Reentrenando con las variables originales (antes del PCA)
rf_perfil = RandomForestClassifier(n_estimators=100, random_state=42)
rf_perfil.fit(X_train_scaled, y_train)

# Extraer las variables más importantes de Random Forest
importancias = pd.DataFrame({
    'Variable': X_encoded.columns,
    'Importancia': rf_perfil.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\n15 VARIABLES MÁS IMPORTANTES (Random Forest)")
print(importancias.head(15).round(2))


#Perfiles combinando ambos algortimos
print("Perfil de clientes")

#Regresión logitica
coeficientes_lr = pd.DataFrame({'Variable': X_encoded.columns,
    'Coeficiente_LR': importancia_lr}).sort_values('Coeficiente_LR', key=abs, ascending=False)

#Random Forest
importancias_rf = pd.DataFrame({'Variable': X_encoded.columns,
    'Importancia_RF': rf_perfil.feature_importances_})

#Unir las variables importantes de ambos algoritmos en una tabla
perfil_completo = coeficientes_lr.merge(importancias_rf, on='Variable')
perfil_completo = perfil_completo.sort_values('Importancia_RF', ascending=False)

print("Tabla de las 15 variables más relevantes según ambos modelos supervisados")
print(perfil_completo.head(15).round(2))


#Contrucción de los perfiles
buenas_caracteristicas = (coeficientes[coeficientes['Coeficiente'] > 0].sort_values(by='Coeficiente', ascending=False).head(5))

# Mal cliente: coeficientes más negativos
malas_caracteristicas = ( coeficientes[coeficientes['Coeficiente'] < 0].sort_values(by='Coeficiente').head(5))

print("Perfil de cliente con bajo riesgo")
for variable in buenas_caracteristicas['Variable']:
    print(variable)

print("\nPerfil de cliente con alto riesgo")
for variable in malas_caracteristicas['Variable']:
    print(variable) 


