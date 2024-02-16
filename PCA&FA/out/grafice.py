import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap

def plot_varianta(alpha, criterii,procent_minimal=80,eticheta_x="Componenta"):
    fig=plt.figure("Plot varianta",figsize=(7,5))
    assert isinstance(fig,plt.Figure)
    ax=fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title("Plot varianta", fontdict={"fontsize":16,"color":"b"})
    ax.set_xlabel("Componenta")
    ax.set_ylabel("Varianta")
    x=np.arange(1,len(alpha)+1)
    ax.set_xticks(x)
    ax.plot(x,alpha)
    ax.scatter(x,alpha,c="r",alpha=0.5)
    ax.axhline(alpha[criterii[0]-1],c="m",label="Criteriul acoperirii minimale("+str(procent_minimal)+"%)")
    if not np.isnan(criterii[1]):
        ax.axhline(1,c="c",label="Criteriul Kaiser")
    if not np.isnan(criterii[2]):
        ax.axhline(alpha[criterii[2]-1],c="g",label="Criteriul Cattell (elbow)")
    ax.legend()


def corelograma(tr,vmin=1,cmap="RdYlBu",annot=True, titlu="Corelograma corelatii factoriale"):
    fig=plt.figure(titlu,figsize=(20,10))
    assert isinstance(fig,plt.Figure)
    ax=fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title(titlu,fontdict={"fontsize":16,"color":"b"})
    heatmap(tr,vmin=vmin, vmax=1, cmap=cmap, annot=annot,ax=ax, annot_kws={"size":7})


def scatter(t,col1='C1',col2="C2",titlu="Plot scoruri"):
    fig=plt.figure(figsize=(8,10))
    assert isinstance(fig,plt.Figure)
    ax=fig.add_subplot(1,1,1,aspect=1)
    assert isinstance(ax,plt.Axes)
    ax.set_xlabel(col1,fontsize=14)
    ax.set_ylabel(col2,fontsize=14)
    ax.set_title(titlu,fontdict={"fontsize":16,"color":"b"})
    ax.scatter(t[col1],t[col2],c="b",alpha=0.5)
    ax.axhline(0)
    ax.axhline(0)
    n=len(t)
    if n<50:
        for i in range(n):
            ax.text(t[col1].iloc[i],t[col2].iloc[i],t.index[i])
    plt.show()


def cerculCorelatiilor(matrice=None, raza=1, k1=0, k2=1, dec=2, valMin=-1, valMax=1,
        etichetaX=None, etichetaY=None, titlu='Cercul corelatiilor'):
    plt.figure(titlu, figsize=(8, 8))
    plt.title(titlu, fontsize=14, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t)*raza for t in T]
    Y = [np.sin(t)*raza for t in T]
    plt.plot(X, Y)
    plt.axvline(x=0, color='g')
    plt.axhline(y=0, color='g')
    if etichetaX==None or etichetaY==None:
        if isinstance(matrice, pd.DataFrame):
            plt.xlabel(xlabel=matrice.columns[k1], fontsize=12, color='k', verticalalignment='top')
            plt.ylabel(ylabel=matrice.columns[k2], fontsize=12, color='k', verticalalignment='bottom')
        else:
            plt.xlabel(xlabel='Var '+str(k1+1), fontsize=12, color='k', verticalalignment='top')
            plt.ylabel(ylabel='Var '+str(k2+1), fontsize=12, color='k', verticalalignment='bottom')
    else:
        plt.xlabel(xlabel=etichetaX, fontsize=12, color='k', verticalalignment='top')
        plt.ylabel(ylabel=etichetaY, fontsize=12, color='k', verticalalignment='bottom')

    if isinstance(matrice, np.ndarray):
        plt.scatter(x=matrice[:, k1], y=matrice[:, k2], c='r', vmin=valMin, vmax=valMax)
        matrice_rotunjita = np.round(matrice, dec)
        for i in range(matrice.shape[0]):
            plt.text(x=matrice[i, k1], y=matrice[i, k2], s='(' +
                     str(matrice_rotunjita[i, k1]) + ', ' + str(matrice_rotunjita[i, k2]) + ')')

    if isinstance(matrice, pd.DataFrame):
        plt.scatter(x=matrice.iloc[:, k1], y=matrice.iloc[:, k2], c='r', vmin=valMin, vmax=valMax)

        for i in range(matrice.values.shape[0]):
            plt.text(x=matrice.iloc[i, k1], y=matrice.iloc[i, k2], s=matrice.index[i])
    plt.savefig("cerc_corelatii")
    plt.show()
    print("imagine cerc")

def afisare():
    plt.show()




