# U direktoriju PSU_LV/LV2/ nalazi se datoteka mtcars.csv koja sadrži različita mjerenja provedena na 32 automobila (modeli 1973-74). 
# Opis pojedinih varijabli nalazi se u datoteci mtcars_info.txt.
# a) Učitajte datoteku mtcars.csv pomoću:
# data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)
# b) Prikažite ovisnost potrošnje automobila (mpg) o konjskim snagama (hp) pomoću naredbe matplotlib.pyplot.scatter.
# c) Na istom grafu prikažite i informaciju o težini pojedinog vozila (npr. veličina točkice neka bude u skladu sa težinom wt).
# d) Izračunajte minimalne, maksimalne i srednje vrijednosti potrošnje (mpg) automobila.
# e) Ponovite zadatak pod d), ali samo za automobile sa 6 cilindara (cyl).

import numpy as np
import matplotlib.pyplot as plt

# usecols kreira svoju matricu podataka i onda se prema njoj treba referencirati dalje u programu
data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

mpg = data[:,0]
hp =  data[:,3]
wt =  data[:,5]
cyl = data[:,1]

plt.scatter(mpg, hp, s=wt*20, alpha=0.6) # 20 puta uvecala da se vidi razlika
plt.xlabel("potrosnja automobila")
plt.ylabel("konjske snage")
plt.title("Ovisnost potrosnje automobila (mpg) o konjskim snagama (hp)")
plt.show()

print(f"Minimalan mpg: {np.min(mpg)}\nMaksimalan mpg: {np.max(mpg)}\nSrednja vrijednost mpg: {np.mean(mpg):.2f}\n")

mpg_6cly = mpg[cyl==6]
print(f"Minimalan mpg_6cyl: {np.min(mpg_6cly)}\nMaksimalan mpg_6cyl: {np.max(mpg_6cly)}\nSrednja vrijednost mpg_6cyl: {np.mean(mpg_6cly):.2f}\n")
