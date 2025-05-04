import numpy as np
import matplotlib.pyplot as plt

#ucitaj sliku 'tiger.png'
img = plt.imread("tiger.png") 
img = img[:,:,0].copy() 

plt.figure() 
plt.title("Originalna slika")
plt.imshow(img, cmap="gray") 
plt.show()

print(img.dtype)
print(img.shape)

# a) posvijetliti sliku (povećati brightness)
img_bright = img + 0.7
img_bright[img_bright > 1.0] = 1.0 

plt.figure()
plt.title("a) Posvijetljena slika")
plt.imshow(img_bright, cmap="gray")
plt.show()

# b) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
img_rot = np.rot90(img, -1)  # -1 za smjer kazaljke na satu

plt.figure()
plt.title("b) Rotirana za 90 stupnjeva")
plt.imshow(img_rot, cmap="gray")
plt.show()

# c) zrcaliti sliku
plt.figure()
plt.title("c) Zrcaljena slika")
img_mirr = np.fliplr(img)
plt.imshow(img_mirr, cmap="gray")
plt.show()

# d) smanjiti rezoluciju slike x puta (npr. 10 puta)
x = 10
img_lq = img[::x, ::x] # uzme svaki x-ti piksel iz slike

plt.figure()
plt.title(f"d) Smanjena rezolucija (x{x})")
plt.imshow(img_lq, cmap="gray")
plt.show()

# e) prikazati samo drugu četvrtinu slike po širini, a prikazati sliku cijelu po visini; ostali dijelovi slike trebaju biti crni
h, w = img.shape # 640 x 960
new_img = np.zeros_like(img)  # crna slika istih dimenzija
start = w // 4  # 960 // 4 = 240
end = w // 2    # 960 // 2 = 480
new_img[:, start:end] = img[:, start:end]

plt.figure()
plt.title("e) Samo druga cetvrtina slike")
plt.imshow(new_img, cmap="gray")
plt.show()

