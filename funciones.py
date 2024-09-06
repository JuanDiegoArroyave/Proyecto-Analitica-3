# -*- coding: utf-8 -*-
####### prueba
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ### para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 
#Analisis para vars cualitativas
#Funcion para ratio de categorias por variable
def cat_summary(dataframe, col_name, plot=False):
    '''
    Genera un ratio de categorias por variable y su diagrama de frecuencias
    '''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    print('\n')

    #Generacion del grafico
    if plot:
        plt.figure(figsize=(9, 6))
        color = "#003F72"
        sns.countplot(x=dataframe[col_name], data=dataframe, color=color, legend=False)
        plt.xticks(rotation=60, ha='right')
        plt.tick_params(axis='both', which='major', labelsize=9)
        plt.show(block=True)
    print('\n')
    print('-----------------------------------------------------------------------------------------------------------------------------------------')


def bivariado(df, var_cuan):
  import plotly.graph_objects as go
  from plotly.subplots import make_subplots
  import numpy as np
  '''
  realizar graficos bivariado para analizar diferencias estadisticas entre las categorias
  de la var respuesta para la var cuantitativa en cuestion
  '''
  fig = make_subplots(rows=1, cols=2)

  #Agrupacion entre 'Asignacion' y 'var_cuan'
  asignacion_por_variable = df.groupby('Attrition')[var_cuan].mean().sort_values(ascending=True)
  renuncia = df.loc[df['Attrition']==1]
  no_renuncia = df.loc[df['Attrition']==0]


  colors=['coral', 'darkred']

  #Grafico de barras
  fig.add_trace(
      go.Bar(
          x=['Renuncia', 'No renuncia'],
          y=asignacion_por_variable.values,
          name='Bar Chart',
          marker_color=colors,
          text=np.round(asignacion_por_variable.values)),
      row=1, col=1
  )

  #Generacion de boxplot para Attrition=1
  fig.add_trace(
      go.Box(y=renuncia[var_cuan], name='renuncia', marker_color='coral'),
      row=1, col=2
  )

  #Generacion de boxplot para Attrition=0
  fig.add_trace(
      go.Box(y=no_renuncia[var_cuan], name='no_renuncia', marker_color='darkred'),
      row=1, col=2
  )

  #Impresion de figura
  fig.update_layout(title_text=f"Decision de renuncia vs {var_cuan}", template='simple_white')
  fig.show()

####Este archivo contienen funciones utiles a utilizar en diferentes momentos del proyecto

###########Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas

def ejecutar_sql (nombre_archivo, cur):
  sql_file=open(nombre_archivo)
  sql_as_string=sql_file.read()
  sql_file.close
  cur.executescript(sql_as_string)
  
 

def imputar_f (df,list_cat):  
        
    
    df_c=df[list_cat]

    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer( strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)
    imputer_c.get_params()
    imputer_n.get_params()

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)


    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)
    df_c.info()
    df_n.info()

    df =pd.concat([df_n,df_c],axis=1)
    return df


def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


def medir_modelos(modelos, scoring, X, y, cv):
    metric_modelos = pd.DataFrame()
    nombres_modelos = []

    for modelo in modelos:
        # Realiza la validación cruzada y calcula el score según el método de evaluación especificado
        scores = cross_val_score(modelo, X, y, scoring=scoring, cv=cv)
        pdscores = pd.DataFrame(scores)

        # Agrega los scores a la DataFrame
        metric_modelos = pd.concat([metric_modelos, pdscores], axis=1)

        # Guarda el nombre de la clase del modelo para usarlo como nombre de columna
        nombres_modelos.append(type(modelo).__name__)

    # Asigna los nombres de los modelos a las columnas
    metric_modelos.columns = nombres_modelos
    return metric_modelos



def preparar_datos (df):
   
    

    #######Cargar y procesar nuevos datos ######
   
    
    #### Cargar modelo y listas 
    
   
    list_cat=joblib.load("salidas\\list_cat.pkl")
    list_dummies=joblib.load("salidas\\list_dummies.pkl")
    var_names=joblib.load("salidas\\var_names.pkl")
    scaler=joblib.load( "salidas\\scaler.pkl") 

    ####Ejecutar funciones de transformaciones
    
    df=imputar_f(df,list_cat)
    df_dummies=pd.get_dummies(df,columns=list_dummies)
    df_dummies= df_dummies.loc[:,~df_dummies.columns.isin(['perf_2023','EmpID2'])]
    X2=scaler.transform(df_dummies)
    X=pd.DataFrame(X2,columns=df_dummies.columns)
    X=X[var_names]
    
    
    
    
    return X