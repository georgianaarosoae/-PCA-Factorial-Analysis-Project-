from functii import *
from grafice import *
import numpy as np
import pandas as pd

np.set_printoptions(5, 10000, suppress=True)


t = pd.read_csv("populatia_ocupata.csv", index_col=1)
variabile_observate = list(t)[1:]

nan_replace(t)

model_acp=acp(t,variabile_observate)
model_acp.fit()
print("Varianta componentelor principale:")
print(model_acp.alpha)

varianta=model_acp.tabelare_varianta()
print(varianta)

alpha=model_acp.alpha
plot_varianta(alpha, model_acp.criterii)

#Analiza corelatiilor dintre variabilele observate si componente
r=model_acp.r
etichete_componente=list(varianta.index)
tr=pd.DataFrame(r,variabile_observate,etichete_componente)
tr.to_csv("R.csv")
corelograma(tr,titlu="Corelogramă corelații dintre variabilele observate și componente")

#Analiza scorurilor
c=model_acp.c
s=c/np.sqrt(alpha)
ts=pd.DataFrame(s,t.index,etichete_componente)
ts.to_csv("Scoruri.csv")
scatter(ts)
scatter(ts,col1="C1",col2="C2")

#Calculul valorilor cosinus
c2=c*c
cosin=c/np.sqrt(c2.T/np.sum(c2,axis=1)).T
t_cosin=pd.DataFrame(cosin,t.index,etichete_componente)
t_cosin.to_csv("cosin.csv")
cosin_sort=t_cosin.apply(func=lambda x:list(x.index[np.flip(x.argsort())]),axis=1)
cosin_sort.name="Componente"
cosin_sort.to_csv("cosin_sort.csv")

#calcul contributii
contrib=c2*100/np.sum(c2,axis=0)
t_contrib=pd.DataFrame(contrib,t.index,etichete_componente)
t_contrib.to_csv("contrib.csv")
contrib_sort=t_contrib.apply(func=lambda x: pd.Series(list(x.index[np.flip(x.argsort())]),np.arange(1,len(x)+1)),
                             axis=0)
contrib_sort.to_csv("contrib_sort.csv")


#calculul comunalitatilor
r2=r*r
comm=np.cumsum(r2,axis=1)
t_comm=pd.DataFrame(comm,variabile_observate,etichete_componente)
t_comm.to_csv("comm.csv")
corelograma(t_comm,0,"Blues", titlu="Comunalitati", annot=False)

# plot corelații dintre variabilele observate și componente (cercul corelațiilor),
cerculCorelatiilor(matrice=r,titlu='Cercul corelatiilor variabilelor initiale in spatiul C1 si C2.')
afisare()
print(r)




