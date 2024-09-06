# -*- coding: utf-8 -*-

# Librerias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# cargar datos de un archivo de excel
df = pd.read_excel(r'C:\Users\alejo\OneDrive\Documentos\Analítica-3\Repos\Recursos_humanos_git\Proyecto-Analitica-3\data\base_modelo_2.xlsx') # Cambiar ruta a segun
df.head(5)

# Se reemplaza en la var respuesta Attrition los valores de Yes por 1 y NaN por 0
# si renunció es 1, sino 0
df['Attrition'] = df['Attrition'].replace('Yes', 1)
df['Attrition'] = df['Attrition'].fillna(0)

# Ver tipo de columnas
df.info()

# Cambiar formato de variables
df['EmployeeID'] = df['EmployeeID'].astype(str)

# Cantidad de personas que han renunciado
print(df['Attrition'].value_counts())

# Importar paquete de funciones propias
from funciones import bivariado, cat_summary




print('-----------------------------------------------------------------------------------------------------------------------------------------')

#Generacion de graficos para tpdas las vars cualitativas
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
for col in cat_cols:
    cat_summary(df, col, plot=False)

df['EmployeeCount'].value_counts()

df['StandardHours'].value_counts()

# Eliminar EmployeeCount y StandardHours debido a que es el mismo valor para todos los registros
# no aporta valor al analisis ni modelado
df.drop('EmployeeCount', axis=1, inplace=True)
df.drop('StandardHours', axis=1, inplace=True)

# Filtrar df por tipo de variable int64 o float64
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

df[num_cols].describe()

# Mapa de correlaciones para variables cuantitativas
correlation_matrix = df[num_cols].corr()

# Mapa de calor de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Otra visualizacion
correlation_matrix['Attrition'].sort_values(ascending=False)
# la var con mayor correlacion es TotalWorkingYears, es inversamente proporcional en un 16% aprox

# Se procede a realizar un analisis bivariado para analizar diferencias estadisticas entre las categorias
# de la var respuesta para la var cuantitativa en cuestion

#Aplicacion de funcion a las variables cuantitativas
cuan_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]
            and col!='Attrition']
for col in cuan_cols:
    bivariado(df, col)


df.to_excel(r'C:\Users\alejo\OneDrive\Documentos\Analítica-3\Repos\Recursos_humanos_git\Proyecto-Analitica-3\data\base_modelo_final.xlsx', index=False)