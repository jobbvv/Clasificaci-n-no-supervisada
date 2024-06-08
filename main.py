import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo Excel
df = pd.read_excel('dataset2.xlsx')

# Convertir el DataFrame a un array de numpy
X = df.values
# Definir el número de grupos (k)
k = 2  
# Crear el modelo de k-medias
kmeans = KMeans(n_clusters=k,random_state=42)
# Ajustar el modelo a los datos
kmeans.fit(X)
# Obtener las etiquetas de los clusters
labels = kmeans.labels_
# Obtener los centroides de los clusters generados automáticamente por kmeans
centroids = kmeans.cluster_centers_
# Agregar las etiquetas de los clusters al DataFrame original
df['cluster'] = labels

# Visualizar los resultados en un gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Colores para cada cluster
colors = ['r', 'g']
# Graficar cada punto con su color de cluster correspondiente
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50, c=colors[i], label=f'Cluster {i}')

# Graficar los centroides generados por kmeans
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=300, c='black', marker='X', label='Centroides')

# Etiquetas y leyenda
ax.set_xlabel('Velocidad_promedio')
ax.set_ylabel('Numero_paradas')
ax.set_zlabel('Calificacion_usuarios')
ax.legend()

# Mostrar el gráfico
plt.show()

# Guardar el DataFrame con las etiquetas de cluster en un nuevo archivo Excel
df.to_excel('dataset2_resultados.xlsx', index=False)




