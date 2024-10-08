import sqlite3 as sql
import pandas as pd
from sklearn import neighbors
from ipywidgets import interact

# conectarse a base de datos sql
conn = sql.connect(r'C:\Users\amqj1\OneDrive\Escritorio\Codigos JD\Analitica III\Marketing\Proyecto-Analitica-3\2. Data\db_movies')
cur=conn.cursor()



#### ver tablas disponibles en base de datos

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()


# Extraer la tabla ratings2 y movies2
df = pd.read_sql('SELECT * FROM movies_2', conn)
df_movies = df.copy()

# eliminar la columna index
df_movies = df_movies.drop('index', axis=1)
df = df.drop('index', axis=1)

# Eliminar columnas tipo srt
df_movies = df_movies.drop('movie_title', axis=1)

# verificar datos faltantes
df_movies.isnull().sum()
# Eliminar datos faltantes
df_movies = df_movies.dropna()
df = df.dropna()


# Modelo de knn para sistema de recomendación product-product
model = neighbors.NearestNeighbors(n_neighbors=6, metric='cosine') 
model.fit(df_movies)

dist, idlist = model.kneighbors(df_movies)

def MovieRecommender(movie_name = list(df['movie_title'].value_counts().index)):
    movie_list_name = []
    movie_id = df[df['movie_title'] == movie_name].index[0]
    
    # Recomendaciones de películas similares
    for newid in idlist[movie_id]:
        movie_list_name.append(df.loc[newid].movie_title)
    
    # Eliminar la película seleccionada de las recomendaciones
    if movie_name in movie_list_name:
        movie_list_name.remove(movie_name)
    
    return movie_list_name

# Interfaz interactiva
print(interact(MovieRecommender))