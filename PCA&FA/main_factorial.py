from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

from functii import *
from grafice import *

np.set_printoptions(5, 10000, suppress=True)

t = pd.read_csv("populatia_ocupata.csv", index_col=0)
variabile_observate = list(t)[1:]

nan_replace(t)

x = t[variabile_observate].values

# Analiza factorabilitatii
test_bartlett = calculate_bartlett_sphericity(x)
print("Test Bartlett:", test_bartlett)
if test_bartlett[1] > 0.001:
    print("Nu exista factori comuni!")
    exit(0)
kmo = calculate_kmo(x)
# print(kmo)
t_kmo = pd.DataFrame(
    {
        "Index KMO": np.append(kmo[0], kmo[1])
    }, variabile_observate + ["KMO Global"]
)
t_kmo.to_csv("Indecsi_kmo.csv")
#corelograma(t_kmo, 0, "Greens", titlu="Index KMO")



# Creare model
n, m = x.shape
#model_fact = FactorAnalyzer(m, rotation=None)
model_fact = FactorAnalyzer(m, rotation="varimax")
model_fact.fit(x)



# Analiza variantei
varianta = model_fact.get_factor_variance()
# print(varianta)
etichete_factori = ["F" + str(i) for i in range(1, m + 1)]
t_varianta = pd.DataFrame(
    {
        "Varianta": varianta[0],
        "Varianta cumulata": np.cumsum(varianta[0]),
        "Procent varianta": varianta[1] * 100,
        "Procent cumulat": varianta[2] * 100
    }, etichete_factori
)
t_varianta.to_csv("Varianta_factori.csv")
alpha = varianta[0]
criterii = criterii_factori(alpha)
print("Numar factori conform criteriilor:", criterii)
plot_varianta(alpha, criterii, 70, eticheta_x="Factor")



# Preluare corelatii variabile-factori
l = model_fact.loadings_
t_l = pd.DataFrame(l, variabile_observate, etichete_factori)
t_l.to_csv("corelatii_variabile_factori.csv")
corelograma(t_l)
#scatter(t_l, col1="F1", col2="F2", titlu="Plot corelatii factoriale")
#scatter(t_l,"F3","F4",titlu="Plot corelatii factoriale")



# Calcul scoruri
f = model_fact.transform(x)
t_f = pd.DataFrame(f, t.index, etichete_factori)
t_f.to_csv("Scoruri_factoriale.csv")
scatter(t_f, col1="F1", col2="F2",titlu="Plot scoruri factoriale")
#scatter(t_f,"F3","F4",titlu="Plot scoruri factoriale")



# Preluare comunalitati
h = model_fact.get_communalities()
t_h = pd.Series(h,variabile_observate)
t_h.name = "Comunalitati"
t_h.to_csv("Comunalitati_AF.csv")
corelograma( pd.DataFrame(t_h),vmin=0,cmap="Blues",titlu="Comunalitati")



# Preluare varianta factori specifici

psi = model_fact.get_uniquenesses()
t_psi = pd.Series(psi,variabile_observate)
t_psi.name="Varianta specifica"
t_psi.to_csv("Varianta_specifica.csv")
corelograma(pd.DataFrame(t_psi),0,"Reds",titlu="Varianta specifica")

afisare()
