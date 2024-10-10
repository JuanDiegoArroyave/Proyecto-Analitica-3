import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Conexión a la base de datos
conn = sql.connect(r'C:\Users\amqj1\OneDrive\Escritorio\Codigos JD\Analitica III\Marketing\Proyecto-Analitica-3\2. Data\db_movies')
cur=conn.cursor()

cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall()
# Carga de datos originales en pandas
ratings = pd.read_sql('SELECT * FROM ratings_2', conn)
ratings

# Pivotar la tabla de ratings para crear una matriz de usuario-libro
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
ratings_matrix
# Entrenamiento y prueba de conjunto
X_train, X_test = train_test_split(ratings_matrix, test_size=0.2, random_state=42)

# Configuración de la grilla de hiperparámetros
param_grid = {
    'n_neighbors': [5, 10, 15, 20],  # número de vecinos
    'metric': ['cosine', 'euclidean']  # métrica a utilizar
}

# Búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(NearestNeighbors(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train)

# Resultados de la búsqueda
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # Convertir a puntuación positiva

print(f"Mejores hiperparámetros: {best_params}")
print(f"Mejor puntuación (MSE): {best_score}")

# Utilizar el mejor modelo encontrado
best_model = grid_search.best_estimator_

# Función para obtener recomendaciones por usuario
def obtener_recomendaciones_por_usuario(n_recomend=10):
    recomendaciones = []

    # Recorrer todos los usuarios en el conjunto de entrenamiento
    for user_id in X_train.index:
        user_vector = X_train.loc[user_id].values.reshape(1, -1)
        
        # Encontrar los vecinos más cercanos
        distances, indices = best_model.kneighbors(user_vector, n_neighbors=best_params['n_neighbors'])
        
        # Almacenar las recomendaciones
        for idx in indices.flatten():
            isbn = X_train.columns[idx]
            if isbn not in ratings[ratings['userId'] == user_id]['movieId'].values:  # Verificar si el usuario ya calificó el libro
                recomendaciones.append({'userId': user_id, 'movieId': isbn})

    return pd.DataFrame(recomendaciones)

# Obtener recomendaciones para todos los usuarios
recomendaciones_df = obtener_recomendaciones_por_usuario(n_recomend=10)

# Mostrar las primeras filas del DataFrame de recomendaciones
print(recomendaciones_df.head())

recomendaciones_df

# Función para obtener recomendaciones
def recomendaciones(user_id, n_recomend=10):
    if user_id not in X_train.index:
        return pd.DataFrame()  # Si el usuario no existe en el conjunto de entrenamiento

    user_vector = X_train.loc[user_id].values.reshape(1, -1)
    distances, indices = best_model.kneighbors(user_vector, n_neighbors=n_recomend)

    # Obtener los libros recomendados
    recommended_movies_indices = indices.flatten()
    recommended_movies = X_train.columns[recommended_movies_indices].tolist()
    
    return recommended_movies

# # Obtener recomendaciones para un usuario específico
# user_id = 24  # Cambia este ID según sea necesario
# recommended_movies = recomendaciones(user_id=user_id, n_recomend=10)
# print(f"Recomendaciones para el usuario {user_id}: {recommended_movies}")

# Cierre de conexión a la base de datos
conn.close()