# -*- coding: utf-8 -*-
####### prueba
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ### para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

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


def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
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

# Función para calcular el AUCPR (Área bajo la curva de Precision-Recall)
def calculate_aucpr(y_true, y_scores):
    return average_precision_score(y_true, y_scores)

# Función para realizar el modelo con balanceo (oversampling) y calcular las métricas
def evaluate_model_with_oversampling(X, y, oversample_ratios):
    results = []  # Para almacenar las métricas
    # Dividimos los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Iteramos sobre los diferentes oversampling ratios (usando SMOTE)
    for over_ratio in oversample_ratios:
        try:
            # Definimos el oversampling SMOTE
            oversample = SMOTE(sampling_strategy=over_ratio, random_state=42)

            # Aplicamos SMOTE en los datos de entrenamiento
            X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)

            # Definimos el clasificador RandomForest
            classifier = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            
            # Entrenamos el modelo
            classifier.fit(X_train_resampled, y_train_resampled)

            # Hacemos las predicciones
            y_pred = classifier.predict(X_test)
            y_scores = classifier.predict_proba(X_test)[:, 1]  # Probabilidades para AUCPR

            # Calculamos las métricas
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            aucpr = calculate_aucpr(y_test, y_scores)

            # Guardamos los resultados en un diccionario
            results.append({
                'oversample_ratio': over_ratio,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'aucpr': aucpr
            })

        except ValueError as e:
            print(f"Error with oversample_ratio {over_ratio}: {e}")
            continue

    # Convertimos los resultados en un DataFrame
    results_df = pd.DataFrame(results)
    return results_df

from sklearn.model_selection import GridSearchCV

def evaluate_rf_hyperparameters(X, y, param_grid):
    # Dividimos los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Definimos el modelo base (Random Forest)
    rf = RandomForestClassifier(random_state=42)
    
    # Definimos la búsqueda de hiperparámetros con GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               scoring='average_precision',  # AUCPR como métrica
                               cv=5,  # Validación cruzada con 5 particiones
                               n_jobs=-1,  # Paralelización
                               verbose=2)

    # Ajustamos el modelo con las combinaciones de hiperparámetros
    grid_search.fit(X_train, y_train)

    # Recuperamos el mejor modelo
    best_model = grid_search.best_estimator_
    
    # Hacemos predicciones sobre los datos de prueba
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)[:, 1]

    # Calculamos las métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    aucpr = average_precision_score(y_test, y_scores)

    # Guardamos los resultados
    results = {
        'best_params': grid_search.best_params_,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'aucpr': aucpr
    }
    
    # Convertimos los resultados en un DataFrame
    results_df = pd.DataFrame([results])
    return results_df, best_model