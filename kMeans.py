from typing import Iterable, Tuple
from random import uniform
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from general import eulidianDistance
FLOAT_MAX = sys.float_info.max

### K MEANS IMPLEMENTATION
def kMeans(k:int,df:pd.DataFrame,columns:Iterable ,EPOC:int=20) -> Tuple[list,pd.DataFrame] :
    min_max = {}
    for column in columns:
        min_max[column] = (df[column].min(),df[column].max())
    # cluster_centers = randomClusterCenters(k,min_max)
    cluster_centers = evenlyDistributerClusterCenters(k,min_max)
    for i in tqdm(range(EPOC),'Clustering ... '):
    # for i in range(EPOC):
        # df = assignCluster(cluster_centers,df,columns)
        df = assignClusterI2(cluster_centers,df,columns)
        cluster_centers = calcClusterCenters(df)
    return cluster_centers, df

### FUNCTION TO RETURN RANDOM CENTERS
def randomClusterCenters(k:int,min_max:dict) -> list:
    cluster_centers = []
    for discard in range(k):
        center = []
        for column in min_max.keys():
            center.append(uniform(min_max[column][0],min_max[column][1]))
        cluster_centers.append(center)
    return cluster_centers

### FUNCTION TO RETURN EVENLY DISTRIPUTED CLUSTER CENTERS
def evenlyDistributerClusterCenters(k:int,min_max:dict) -> list:
    cluster_centers = []
    for i in range(k):
        center = []
        for column in min_max.keys():
            width = min_max[column][1]/k
            center.append(uniform(i*width,(i+1)*width))
        cluster_centers.append(center)
    return cluster_centers

### FUNCTION TO ASSIGN DATA POINTS TO CLUSTER CENTERS
def assignCluster(cluster_centers:list,df:pd.DataFrame,columns:Iterable) -> pd.DataFrame:
    if len(cluster_centers[0]) == len(columns):
        df['cluster'] = [None for i in range(len(df[columns[0]]))]
        for index,row in df.iterrows():
            min_distance = FLOAT_MAX
            min_index = None
            for i,center in enumerate(cluster_centers):
                distance = eulidianDistance(center,[row[column] for column in columns])
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            df.at[index,'cluster'] = min_index
    return df

### 2ND ASSIGN CLUSTER IMPLEMENTAION
def assignClusterI2(cluster_centers:list,df:pd.DataFrame,columns:Iterable) -> pd.DataFrame:
    if len(cluster_centers[0]) == len(columns):
        ccdf = pd.DataFrame(cluster_centers,columns=columns)
        for index,row in ccdf.iterrows():
            df[index] = (sum([(row[column]-df[column])**2 for column in columns]))**0.5
        df['distance'] = df[[index for index,row in ccdf.iterrows()]].min(axis=1)
        df['cluster'] = df[[index for index,row in ccdf.iterrows()]].idxmin(axis=1)
    return df

### FUNCTION TO RECALCULATE CLUSTER CENTER
def calcClusterCenters(df:pd.DataFrame) -> list:
    cluster_centers = []
    clusters = np.sort(df['cluster'].unique())
    k = len(clusters)
    for i in range(k):
        temp = df[df.cluster==clusters[i]].head()
        cluster_centers.append([temp.x.mean(),temp.y.mean()])
    return cluster_centers