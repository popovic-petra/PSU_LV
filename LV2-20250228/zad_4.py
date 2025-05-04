import numpy as np
import matplotlib.pyplot as plt

def sahovnica (velicina_kvadrata, broj_redova, broj_stupaca):
    crni = np.zeros((velicina_kvadrata, velicina_kvadrata))
    bijeli = np.ones((velicina_kvadrata, velicina_kvadrata)) * 1.0

    # pravimo jedan red sahovnice
    red = []
    for j in range(broj_stupaca):
        if j % 2 == 0:      # ako je indeks paran dodaj crni
            red.append(crni)
        else:
            red.append(bijeli) # ako je indeks neparan dodaj bijeli
    red = np.hstack(red)    # horizontalno spajanje kvadrata u jedan red

    # Napravi cijelu sliku slaganjem redova naizmjenično
    slika = []
    for i in range(broj_redova):
        if i % 2 == 0:
            slika.append(red)
        else:
            slika.append(np.hstack([bijeli if j % 2 == 0 else crni for j in range(broj_stupaca)]))
    slika = np.vstack(slika)  # vertikalno spajanje redova

    return slika

img = sahovnica(velicina_kvadrata=50, broj_redova=4, broj_stupaca=5)

plt.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("Sahovnica")
plt.show()