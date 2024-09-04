# -*- coding: utf-8 -*-

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