# Modelo hibrido
'''
Si un usuario no ha visto ningun película, tiene la opcion de elegir una de interés para recibir recomendaciones (modelo product)
Si el usuario ya ha puntuado y visto peliculas, aplica el modelo user

'''


import pandas as pd
from user import recomendaciones
import sqlite3 as sql
from product import MovieRecommender


# Borrar elementos de consola (para consola interactiva)

# Limpia consola tradicional
import os
os.system('cls')



# conectarse a base de datos sql
conn = sql.connect(r'C:\Users\amqj1\OneDrive\Escritorio\Codigos JD\Analitica III\Marketing\Proyecto-Analitica-3\2. Data\db_movies')
cur=conn.cursor()


# Extraer la tabla ratings2 y movies2
df_movies = pd.read_sql('SELECT * FROM movies_2', conn)
df_ratings = pd.read_sql('SELECT * FROM ratings_2', conn)



# Obtener recomendaciones para un usuario específico
user_id = 24  # Cambia este ID según sea necesario
recommended_movies = recomendaciones(user_id=user_id, n_recomend=10)
print(f"Recomendaciones para el usuario {user_id}: {recommended_movies}")


aux = pd.DataFrame(df_ratings['movieId'].value_counts())

# Peliculas con mas de 50 valoraciones
aux = aux[aux['count'] >= 50]

mejores_movies = df_ratings[df_ratings['movieId'].isin(aux.index)]     

mejores_ratings = mejores_movies.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(5)


# Si los usuarios son nuevos, no aparecen aun en df_ratings
def hybrid(userid):
    # Si el usuario no ha visto ninguna película, se recomienda la más popular
    if userid not in df_ratings['userId'].unique():
        print('Puede que te interese las siguientes peliculas: ')
        for i in mejores_ratings.index:
            print(df_movies[df_movies['movieId'] == i]['movie_title'].iloc[0])

        
    # Si el usuario ya ha puntuado y visto peliculas, se aplica el modelo user
    else:
        recom = recomendaciones(user_id=userid, n_recomend=10)
        recom_titles = df_movies[df_movies['movieId'].isin(recom)]['movie_title'].unique().tolist()
        print("Te recomendamos las siguientes peliculas: ")
        for i in recom_titles:
            print(i)
