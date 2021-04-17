# -*- coding: utf-8 -*-


import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch

from sklearn.cluster import estimate_bandwidth

from sklearn import metrics
from math import floor
import seaborn as sns

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

datos = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

'''
for col in datos:
   missing_count = sum(pd.isnull(datos[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

#Se pueden reemplazar los valores desconocidos por un número
#datos = datos.replace(np.NaN,0)

#O imputar, por ejemplo con la media      
for col in datos:
   datos[col].fillna(datos[col].mean(), inplace=True)
 

     
#seleccionar casos
caso1 = datos.loc[(datos['EC']==1)]
caso2 = datos.loc[(datos['SEXO']==2) & (datos['COLEGIO']==1) & (datos['JUGAR']==1) & (datos['ROPA']==1)]
caso3 = datos.loc[(datos['RELIGION']==1) | (datos['RELIGION']==3)]

casos = [caso1, caso2, caso3]

#seleccionar variables de interés para clustering
usadas1 = ['INGREHOG', 'EDADESTABLE', 'NHIJOS', 'ESTUDIOSA', 'NEMBTRAREPRO']
usadas2 = ['ESTUDIOSA', 'NEMPLEOS', 'NHIJOS', 'INGREHOG_INTER', 'ESTUPAREJA', 'NPARANT', ]
usadas3 = ['TEMPRELA', 'NHIJOSDESEO', 'NHIJOS', 'PRACTICANTE','EDADHIJO1', 'MAMPRIMHIJO', 'ANON']

usadas = [usadas1,usadas2,usadas3]


X = caso3[usadas3]
X_normal = X.apply(norm_to_zero_one)

'''

for i in range(0,3):
    caso = casos[i]
    usadasC = usadas[i]
    X = caso[usadasC]
    X_normal = X.apply(norm_to_zero_one)
    
'''


#==============================================================================
#====================================KMEDIAS===================================
#==============================================================================

for i in range(2,20):

    print('----- Ejecutando k-Means, k_'+str(i)+":",end='')
    k_means = KMeans(init='k-means++', n_clusters=i, n_init=5)
    t = time.time()
    cluster_predict = k_means.fit_predict(X_normal) 
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end='')
    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')

#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster:")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))

    centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
    centers_desnormal = centers.copy()

#se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    hm = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    hm.figure.savefig("heatmapKMeans_"+str(i)+".png")

#----------------------------------------------------------VISUALIZACIÓN

#
    print("---------- Preparando el scatter matrix...")

#se añade la asignación de clusters como columna a X
    X_kmeans = pd.concat([X, clusters], axis=1)
    sns.set()
    variables = list(X_kmeans)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("kmeans_"+str(i)+".png")
    print("")
 
#


#==============================================================================
#================================MEANSHIFT=====================================
#==============================================================================


#******************************************** //NO PARAM***********


print('----- Ejecutando MeanShift__NOPARAM',end='')
m_shift = MeanShift()
        
t2 = time.time()

cluster_predict2 = m_shift.fit_predict(X_normal)
    
tiempo2 = time.time() - t2
print(": {:.2f} segundos, ".format(tiempo2), end='')
metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict2)
print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')

#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
   
metric_SC = metrics.silhouette_score(X_normal, cluster_predict2, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
clusters2 = pd.DataFrame(cluster_predict2,index=X.index,columns=['cluster'])

print("Tamaño de cada cluster:")
size=clusters2['cluster'].value_counts()
for num,i in size.iteritems():
    print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters2)))

centers2 = pd.DataFrame(m_shift.cluster_centers_,columns=list(X))
centers_desnormal2 = centers2.copy()

#se convierten los centros a los rangos originales antes de normalizar
for var in list(centers2):
    centers_desnormal2[var] = X[var].min() + centers2[var] * (X[var].max() - X[var].min())

hm2 = sns.heatmap(centers2, cmap="YlGnBu", annot=centers_desnormal2, fmt='.3f')
hm2.figure.savefig("heatmapMeanShift__"+str(i)+"__NOPARAM.png")

print("---------- Preparando el scatter matrix...")

#se añade la asignación de clusters como columna a X
M_Shift = pd.concat([X, clusters2], axis=1)
sns.set()
variables = list(M_Shift)
variables.remove('cluster')
sns_plot = sns.pairplot(M_Shift, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
sns_plot.savefig("mshiftscatter__"+str(i)+"__NOPARAM.png")
print("")




#*******************************************************************

for i in [True,False]:

    print('----- Ejecutando MeanShift__clusterall='+str(i)+"__SINBW",end='')
    m_shift = MeanShift(cluster_all=i)
        
    t2 = time.time()

    cluster_predict2 = m_shift.fit_predict(X_normal)
    
    tiempo2 = time.time() - t2

    print(": {:.2f} segundos, ".format(tiempo2), end='')
    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict2)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')

#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict2, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
    clusters2 = pd.DataFrame(cluster_predict2,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster:")
    size=clusters2['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters2)))

    centers2 = pd.DataFrame(m_shift.cluster_centers_,columns=list(X))
    centers_desnormal2 = centers2.copy()

#se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers2):
        centers_desnormal2[var] = X[var].min() + centers2[var] * (X[var].max() - X[var].min())

    hm2 = sns.heatmap(centers2, cmap="YlGnBu", annot=centers_desnormal2, fmt='.3f')
    hm2.figure.savefig("heatmapMeanShift__"+str(i)+"__SINBW.png")

#----------------------------------------------------------VISUALIZACIÓN

#
    print("---------- Preparando el scatter matrix...")

#se añade la asignación de clusters como columna a X
    M_Shift = pd.concat([X, clusters2], axis=1)
    sns.set()
    variables = list(M_Shift)
    variables.remove('cluster')
    sns_plot = sns.pairplot(M_Shift, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("mshiftscatter__"+str(i)+"__SINBW.png")
    print("")

#==============================================================================



#==============================================================================
#================================GAUSSIANM=====================================
#==============================================================================


for i in range(2,20):   
    print('----- Ejecutando GaussianMixture_'+str(i),end='')
    
    gaussian = GaussianMixture(n_components=i)

    t3 = time.time()

    cluster_predict3 = gaussian.fit_predict(X_normal)
    
    tiempo3 = time.time() - t3

    print(": {:.2f} segundos, ".format(tiempo3), end='')
    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict3)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')

#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict3, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
    clusters3 = pd.DataFrame(cluster_predict3,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster:")
    size=clusters3['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters3)))

    centers3 = pd.DataFrame(gaussian.means_,columns=list(X))
    centers_desnormal3 = centers3.copy()

#se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers3):
        centers_desnormal3[var] = X[var].min() + centers3[var] * (X[var].max() - X[var].min())

    hm3 = sns.heatmap(centers3, cmap="YlGnBu", annot=centers_desnormal3, fmt='.3f')
    hm3.figure.savefig("heatmapGaussianMixture_"+str(i)+".png")

#----------------------------------------------------------VISUALIZACIÓN

#
    print("---------- Preparando el scatter matrix...")
    
#se añade la asignación de clusters como columna a X
    GM = pd.concat([X, clusters3], axis=1)
    sns.set()
    variables = list(GM)
    variables.remove('cluster')
    sns_plot = sns.pairplot(GM, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("GaussianMixtureScatter"+str(i)+".png")
    print("")
#'''




#==============================================================================
#===============================MINIBATCH KMEANS===============================
#==============================================================================
for i in range(2,20):

    print('----- Ejecutando MiniBatch KMeans n_'+str(i),end='')
    ap = MiniBatchKMeans(n_clusters=i)

    t4 = time.time()

    cluster_predict4 = ap.fit_predict(X_normal)

    tiempo4 = time.time() - t4

    print(": {:.2f} segundos, ".format(tiempo4), end='')
    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict4)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')

#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict4, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
    clusters4 = pd.DataFrame(cluster_predict4,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster:")
    size=clusters4['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters4)))

    centers4 = pd.DataFrame(ap.cluster_centers_ ,columns=list(X))
    centers_desnormal4 = centers4.copy()

#se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers4):
        centers_desnormal4[var] = X[var].min() + centers4[var] * (X[var].max() - X[var].min())

    hm4 = sns.heatmap(centers4, cmap="YlGnBu", annot=centers_desnormal4, fmt='.3f')
    hm4.figure.savefig("heatmapMiniBatchKMeans_"+str(i)+".png")

#----------------------------------------------------------VISUALIZACIÓN

#
    print("---------- Preparando el scatter matrix...")

#se añade la asignación de clusters como columna a X
    DBS = pd.concat([X, clusters4], axis=1)
    sns.set()
    variables = list(DBS)
    variables.remove('cluster')
    sns_plot = sns.pairplot(DBS, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("MiniBatchKMEanscatter_"+str(i)+".png")
    print("")
#'''



#==============================================================================
#====================================BIRCH=====================================
#==============================================================================

for i in [0.05,0.08,0.09,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
 
    print('----- Ejecutando Birch_'+str(i),end='')
    birch = Birch(threshold=i)

    t5 = time.time()

    cluster_predict5 = birch.fit_predict(X_normal)

    tiempo5 = time.time() - t5

    print(": {:.2f} segundos, ".format(tiempo5), end='')
    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict5)
    print("Calinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')

#el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
   
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict5, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))

#se convierte la asignación de clusters a DataFrame
    clusters5 = pd.DataFrame(cluster_predict5,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster:")
    size=clusters5['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters5)))

    centers5 = pd.DataFrame(birch.subcluster_centers_,columns=list(X))
    centers_desnormal5 = centers5.copy()

#se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers5):
        centers_desnormal5[var] = X[var].min() + centers5[var] * (X[var].max() - X[var].min())

    hm5 = sns.heatmap(centers5, cmap="YlGnBu", annot=centers_desnormal5, fmt='.3f')
    hm5.figure.savefig("heatmapBirch__"+str(i)+".png")

#----------------------------------------------------------VISUALIZACIÓN

#
    print("---------- Preparando el scatter matrix...")

#se añade la asignación de clusters como columna a X
    bch = pd.concat([X, clusters5], axis=1)
    sns.set()
    variables = list(bch)
    variables.remove('cluster')
    sns_plot = sns.pairplot(bch, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig("birchscatter__"+str(i)+".png")
    print("")
#'''

#==============================================================================


