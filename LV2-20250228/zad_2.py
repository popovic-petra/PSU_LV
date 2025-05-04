import numpy as np
import matplotlib.pyplot as plt

#a) ucitajte datoteku mtcars.csv
data = np.loadtxt(open("mtcars.csv", "rb"), 
                  usecols=(1,2,3,4,5,6), 
                  delimiter=",", 
                  skiprows=1)

mpg = data[:, 0]     # potrosnja
cyl = data[:, 1]     # broj cilindara
hp = data[:, 3]      # konjske snage
wt = data[:, 5]      # masa

#b) prikazi ovisnogst mpg o hp pomocu matplotlib.pyplot.scatter
plt.figure(figsize=(10, 6))

#c) size markera (s) predstavlja masu vozila
plt.scatter(hp, mpg, s=wt * 50, c='blue', alpha=0.6)
plt.title("Ovisnost potrosnje o konjskoj snazi")
plt.xlabel("Konjske snage (hp)")
plt.ylabel("Potrosnja (mpg)")

plt.show()

#d) statistika za sve automobile
print("Statistika potrosnje (mpg) za sve automobile:")
print(f"Minimalno: {np.min(mpg)}")
print(f"Maksimalno: {np.max(mpg)}")
print(f"Srednja vrijednost: {np.mean(mpg):.2f}")

#e) statistika za automobile sa 6 cilindara
mpg_6cyl = mpg[cyl == 6]
print("Statistika potrosnje (mpg) za automobile sa 6 cilindara:")
print(f"Minimalno: {np.min(mpg_6cyl)}")
print(f"Maksimalno: {np.max(mpg_6cyl)}")
print(f"Srednja vrijednost: {np.mean(mpg_6cyl):.2f}")

