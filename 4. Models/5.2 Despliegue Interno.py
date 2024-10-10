# Script de cara al usuario donde puede ingresar su id para recomendacion
from hibrido import hybrid
import pandas as pd
import sqlite3 as sql

conn=sql.connect(r'C:\Users\Estefanìa\OneDrive\Escritorio\Python\db_movies') ### crear cuando no existe el nombre de cd  y para conectarse cuando sí existe.
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

df_id=pd.read_sql("select distinct userId from ratings_2", conn)
df_id = df_id.reset_index(drop = True)

df = {}


for id in df_id['userId']: 

    user = id
    a = hybrid(userid=user)
    df[id] = a

df_pred = pd.DataFrame.from_dict(df, orient='index')