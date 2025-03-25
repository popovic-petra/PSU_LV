# Na temelju primjera 2.5. učitajte sliku 'tiger.png'. Manipulacijom odgovarajuće numpy matrice pokušajte:
# a) posvijetliti sliku (povećati brightness),
# b) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
# c) zrcaliti sliku,
# d) smanjiti rezoluciju slike x puta (npr. 10 puta),
# e) prikazati samo drugu četvrtinu slike po širini, a prikazati sliku cijelu po visini; ostali dijelovi slike trebaju biti crni.

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")
img = img[:,:,0].copy()

print(img.shape)
print(img.dtype)

plt.figure() # stvara novi prozor
plt.title("Normalna slika") 
plt.imshow(img, cmap="gray")

# a) posvijetljena slika
plt.figure() # stvara novi prozor 
bright_img = np.clip(img + 0.9, 0, 1)  # s obzirom da je dtype float32, raspon "boja" je od 0 do 1, a na svaki piksel dodamo 0.2
plt.title("Posvijetljena slika") 
plt.imshow(bright_img, cmap="gray")

# b) rotirana slika
plt.figure() # stvara drugi prozor 
rotated_img = np.rot90(img, k=-1) # k=-1 za rotaciju u smjeru kazaljke jer po defaultu ide u smjeru obrnutom od kazaljke na satu
plt.title("Rotirana slika") 
plt.imshow(rotated_img, cmap="gray")

# c) zrcaljena slika
plt.figure() # stvara drugi prozor 
mirrored_img = np.fliplr(img)
plt.title("Zrcaljena slika") 
plt.imshow(mirrored_img, cmap="gray")

# d) slika smanjene rezolucije
plt.figure() # stvara drugi prozor 
low_res_img = img[::10, ::10]
plt.title("Slika smanjene rezolucije") 
plt.imshow(low_res_img, cmap="gray")

# e) druga cetvrtine slike po sirini
plt.figure()
masked_img = np.zeros_like(img)
height, width = img.shape

# izdvajamo drugu cetvrtinu po sirini, a cijelu visinu
masked_img[:, width//4:width//2] = img[:, width//4:width//2]
plt.title("Slika druge cetvrtine slike po sirini") 
plt.imshow(masked_img, cmap='gray')

plt.show()