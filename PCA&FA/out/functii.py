import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace(t):
    assert isinstance(t,pd.DataFrame)
    for v in t.columns:
        if t[v].isna().any():
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(),inplace=True)
            else:
                t[v].fillna(t[v].mode()[0],inplace=True)

class acp():
    def __init__(self,t,variabile_observate):
        assert isinstance(t,pd.DataFrame)
        self._x=t[variabile_observate].values
        self.variabile_observate=variabile_observate

    @property
    def x(self):
        return self._x

    def fit(self,std=True,nlib=0,procent_minim_varianta=80):
        n,m=self._x.shape
        x_=self._x-np.mean(self._x,axis=0)
        if std:
            x_=x_/np.std(self._x,axis=0)
        r_v=(1/(n-nlib))*x_.T@x_
        valp, vecp = np.linalg.eig(r_v)
        k = np.flip(np.argsort(valp))
        self._alpha = valp[k]
        self._a = vecp[:, k]
        self._c = x_ @ self._a
        procent_cumulat = np.cumsum(self.alpha) * 100 / sum(self._alpha)

        k1 = np.where(procent_cumulat > procent_minim_varianta)[0][0] + 1
        if std:
            k2 = len(np.where(self._alpha > 1)[0])
        else:
            k2 = np.NAN


        eps=self.alpha[:m-1]-self.alpha[1:]
        sigma=eps[:m-2]-eps[1:]
        negative=np.where(sigma<0)[0]
        if len(negative)>0:
            k3=negative[0]+2
        else:
            k3=np.NAN
        self.criterii=(k1,k2,k3)
        if std:
            self.r=self._a*np.sqrt(self.alpha)
        else:
            self.r=np.corrcoef(x_,self._c,rowvar=False)[:m,m:]
    @property
    def alpha(self):
        return self._alpha

    @property
    def a(self):
        return self._a

    def tabelare_varianta(self):
        varianta_cumulata = np.cumsum(self._alpha)
        procent_varianta = self._alpha * 100 / sum(self._alpha)
        procent_cumulat = np.cumsum(procent_varianta)
        return pd.DataFrame(
            {
                "Varianta": self.alpha ,
                "Varianta cumulata":varianta_cumulata,
                "Procent varianta": procent_varianta,
                "Procent cumulat":procent_cumulat
            },["C"+str(j) for j in range (1,len(self._alpha)+1)]
        )
    @property
    def c(self):
        return self._c


def criterii_factori(alpha, procent_minim_varianta=70):
    m = len(alpha)
    procent_cumulat = np.cumsum(alpha) * 100 / m
    k1 = np.where(procent_cumulat > procent_minim_varianta)[0][0] + 1
    k2 = len(np.where(alpha > 1)[0])
    eps = alpha[:m - 1] - alpha[1:]
    sigma = eps[:m - 2] - eps[1:]
    negative = np.where(sigma < 0)[0]
    if len(negative) > 0:
        k3 = negative[0] + 2
    else:
        k3 = np.NAN
    return k1, k2, k3

