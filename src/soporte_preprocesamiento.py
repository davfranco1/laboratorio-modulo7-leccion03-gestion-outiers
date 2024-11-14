# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import math

# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest

# Librerías para graficar
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from itertools import product, combinations

from tqdm import tqdm

# ------------- #

def gestion_nulos_lof(df, col_numericas, list_neighbors, lista_contaminacion):
    
    combinaciones = list(product(list_neighbors, lista_contaminacion))
    
    for neighbors, contaminacion in tqdm(combinaciones):
        lof = LocalOutlierFactor(n_neighbors=neighbors, 
                                 contamination=contaminacion,
                                 n_jobs=-1)
        df[f"outliers_lof_{neighbors}_{contaminacion}"] = lof.fit_predict(df[col_numericas])

    return df

def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    for categoria in dataframe[columna_control].unique():
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe(include = "O").T)
        
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe().T)


def separar_dataframe(dataframe):
    return dataframe.select_dtypes(include = np.number), dataframe.select_dtypes(include = "O")

def plot_numericas(dataframe):
    cols_numericas = dataframe.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize= (15,10))
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        sns.histplot(x = columna, data= dataframe, ax = axes[indice])
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()


def plot_cat(dataframe, paleta="mako", tamano_grafica=(15,10)):
    cols_categoricas = dataframe.columns
    num_filas = math.ceil(len(cols_categoricas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize= tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        sns.countplot(x = columna,
                     data= dataframe,
                     ax = axes[indice],
                     palette=paleta,
                     order = dataframe[columna].value_counts().index)
        
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")
        axes[indice].tick_params(rotation = 90)

    if len(cols_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()

def relacion_vs_cat(dataframe, variable_respuesta, paleta="mako", tamano_grafica=(15,10)):
    df_cat = separar_dataframe(dataframe)[1]
    cols_categoricas = df_cat.columns
    num_filas = math.ceil(len(cols_categoricas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize= tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        datos_agrupados = dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta, ascending = False)
        display(datos_agrupados)
        sns.barplot(x= columna,
                    y= variable_respuesta,
                    data= datos_agrupados,
                    ax = axes[indice],
                    palette=paleta)
        
        axes[indice].tick_params(rotation = 90)
        axes[indice].set_title(f"Relación entre {columna} y {variable_respuesta}")
        axes[indice].set_xlabel
    
    plt.tight_layout()


def relacion_vs_numericas(dataframe, variable_respuesta, paleta="mako", tamano_grafica=(15,10)):
    numericas = separar_dataframe(dataframe)[0]
    cols_numericas = numericas.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize= tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        if columna == variable_respuesta:
            fig.delaxes(axes[indice])
            pass
        else:
            sns.scatterplot(
                x= columna,
                y = variable_respuesta,
                data = numericas,
                ax = axes[indice]
            )
    
    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()


def matriz_correlacion(dataframe):
    
    matriz_corr = dataframe.corr(numeric_only=True)

    plt.figure(figsize=(10, 5))
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))

    # Plot the heatmap
    sns.heatmap(matriz_corr,
                annot=True,
                vmin=-1,
                vmax=1,
                mask=mascara,
                cmap="coolwarm")
    plt.show()


def detectar_outliers(dataframe, color="orange", tamano_grafica=(15,10)):
    df_num = separar_dataframe(dataframe)[0]

    num_filas = math.ceil(len(df_num.columns) / 2)

    fig, axes = plt.subplots(ncols=2, nrows=num_filas, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):
        sns.boxplot(
            x=columna,
            data=df_num,
            ax=axes[indice],
            color=color,
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5}
        )
        
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")

    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()
    plt.show()

#plt.xlim([-1,1]) limita los valores de los ejes.