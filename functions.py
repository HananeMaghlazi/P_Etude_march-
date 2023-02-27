# Import des librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import plotly.express as px

# Pour les warnings
import warnings

warnings.filterwarnings("ignore")
# Pour les stats
from scipy import stats

# Pour la modélisation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


def colonne(df):
    """Itération sur les colonnes du dataframe pour afficher le nombre unique des valeurs dans chaque colonne
    Exploration des colonnes"""
    for col in df.columns:
        print("La colonne ", col, " : contient", df[col].nunique(), "valeur unique")


def format_pourcentage(value):
    '''
    Format a percentage with 1 digit after comma 
    '''
    return "{0:.4f}%".format(value * 100)


def missing_data(df):
    """fonction qui retourne le nombre de nan total dans un df"""
    return df.isna().sum().sum()



def missing_percent(df):
    """ fonction qui retourne le nombre de nan total dans un df en pourcentage"""
    return df.isna().sum().sum() / (df.size)



def summary(df):
        """"Fonction summary du dataframe elle affciche la taille du df,nbre unique de la variable,nana et valeur minimale"""
        obs = df.shape[0]
        types = df.dtypes
        counts = df.apply(lambda x: x.count())
        #min = df.min()
        uniques = df.apply(lambda x: x.unique().shape[0])
        nulls = df.apply(lambda x: x.isnull().sum())
        print("Data shape:", df.shape)
        # cols = ["types", "counts", "uniques", "nulls","min","max"]
        cols = ["types", "counts", "uniques", "nulls"]
        str = pd.concat([types, counts, uniques, nulls], axis=1, sort=True)

        str.columns = cols
        dtypes = str.types.value_counts()
        print("___________________________\nData types:")
        print(str.types.value_counts())
        print("___________________________")
        return str



def missing_values(df):
    """ Fonction qui retourne un df avec nombre de nan et pourcentage"""
    nan = pd.DataFrame(columns=["Variable", "nan", "%nan"])
    nan["Variable"] = df.columns
    missing = []
    percent_missing = []
    for col in df.columns:
        nb_missing = missing_data(df[col])
        pc_missing = format_pourcentage(missing_percent(df[col]))
        missing.append(nb_missing)
        percent_missing.append(pc_missing)
    nan["nan"] = missing
    nan["%nan"] = percent_missing
    return nan.sort_values(by="%nan", ascending=False)





def plot_nan(df):
    """ Fonction nan et plot"""
    fig = plt.figure(figsize=(22, 10))

    nan_p = df.isnull().sum().sum() / len(df) / len(df.columns) * 100
    plt.axhline(y=nan_p, linestyle="--", lw=2)
    plt.legend(["{:.2f}% Taux global de nan".format(nan_p)], fontsize=14)

    null = df.isnull().sum(axis=0).sort_values() / len(df) * 100
    sns.barplot(x=null.index, y=null.values)

    plt.ylabel("%")
    plt.title("Pourcentage de NAN pour chaque variable")
    plt.xticks(rotation=70)
    plt.show()
    

def plot_remp(df):
    """  Fonction remplissage et plot"""
    remplissage_df = df.count().sort_values(ascending=True)
    ax = remplissage_df.plot(kind="bar", figsize=(15, 15))
    ax.set_title("Remplissage des données")
    ax.set_ylabel("Nombre de données")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=14)
    plt.tight_layout()




def missing_rows(df):
    """ Fonction de nan par lignes"""
    lines_nan_info = []
    for index, row in df.iterrows():
        lines_nan_info.append((row.isna().sum().sum() / df.shape[1]) * 100)
        df_lines_nan_info = pd.DataFrame(np.array(lines_nan_info), columns=["nan %"])
    return df_lines_nan_info.sort_values(by=["nan %"], ascending=False)



def neg_to_zero(x):
    """ Fonction pour les valeurs en dessous de 0"""
    if x <= 0:
        return 1
    else:
        return x



def correlation_matrix(df):
    """ Affiche la matrice de corrélations"""
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(
        df.corr(),
        mask=mask,
        center=0,
        cmap="Reds",
        linewidths=1,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
    )
    plt.title("Matrice des corrélations", fontsize=15, fontweight="bold")
    plt.show()



def outliers(df, str_columns):
    for i in range(len(str_columns)):
        col = str_columns[i]
        # Calcul du quantile 0,25 qui est le quartile q1 : Calcul de la borne inférieure
        q1 = df[col].quantile(0.25)
        # Calcul du quantile 0,75 qui est le quartile q3 : Calcul de la borne supérieure
        q3 = round(df[col].quantile(0.75), 2)
        # l'écart interquartile (IQR)
        iqr = q3 - q1
        # Mise en évidence de valeurs aberrante faibles
        low = q1 - (1.5 * iqr)
        # Mise en évidence de valeurs aberrante élevées
        high = q3 + (1.5 * iqr)
        # filter the dataset with the IQR
        df_outlier = df.loc[(df[col] > high) | (df[col] < low)]

        return df_outlier


def plot_cnt(df,col,title):
    countplt, ax = plt.subplots(figsize=(15, 10))
    ax = sns.countplot(
    x=col,
    data=df,
    ax=ax,
    order=df[col].value_counts().index,
    )
    
    ax.set_title(title,fontsize=18, color="b", fontweight="bold")
    plt.xticks(rotation=60)
    for rect in ax.patches:
        ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + 0.75,
        rect.get_height(),
        horizontalalignment="center",
        fontsize=11,
        )
        countplt




def heatmap(df):
    """heatmap des données"""
    # the mean value in total
    total_avg = df.iloc[:, 0:8].mean()
    total_avg
    # calculat proportion
    cluster_avg = df.groupby("Cluster").mean()
    prop_rfm = cluster_avg / total_avg
    # heatmap with RFM
    sns.heatmap(prop_rfm, cmap="Oranges", fmt=".2f", annot=True)
    plt.title("Heatmap des clusters")
    plt.plot()





