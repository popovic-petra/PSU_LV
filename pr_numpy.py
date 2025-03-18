# numpy je python biblioteka za numericke proracune
# sadrzi niz funkcija za razlicite numericke rutine
# u odnosu na osnovni python, numpy je bolji zbog efikasnog matricnog racunanja i implementacije polja
# osnovni objekt numpy biblioteke je visedim polje ndarray 
# import numpy as np, pristupa se pomocu np.ime_funkcije

import numpy as np

# print("Matrica a:\n")

a = np.array([1, 2, 3, 5, 7])         # napravi polje od 3 elementa
# print(f"tip polja: {type(a)}")  # ispisuje <class 'numpy.ndarray'>
# print(f"broj elemenata u polju: {a.shape}")          # ispisi koliko elemenata ima polje (3,)
# print(a[1], a[0], a[2])         # ispisi prvi, nulti, drugi element
# a[1] = 9                        # promjeni vrijednost a[1] iz 2 u 9
# print(a)                        # ispisi a
# print(a[1:2])
# print(a[1:-1])

# print("\nMatrica b:\n")

b = np.array([[3,7,1],[4,5,6]])     # napravi 2 dimenzionalno polje (matricu)
# print(b.shape)                      # ispiši dimenzije polja -> (2, 3)
# print(b)                            # ispiši cijelo polje b -> [[3 7 1]\n [4 5 6]]
# print(b[0, 2], b[0, 1], b[1, 1])    # ispiši neke elemente polja -> 1 7 5
# print(b[0:2,0:1])                   # prvi arg 0:2 predstavlja redove, 0:1 stupce tj, 0 i 1 red i 0 stupac, ispisuje kao 2D podniz
# print(b[:,0])                       # ispisi oba reda i nulti stupac -> ispisuje kao 1D polje

# c = np.zeros((4,2))                 # stvara matricu 4x2 sa 0, defaultno uzima float tip podatka
# print(c)
# d = np.ones((3,2), dtype=int)       # stvara matricu 3x2 sa 1
# print(d)
# e = np.full((3,4),5)   
# e = np.full((3,4),5.0)              # stvara matricu 3x4 sa 5, uzima tip koji smo mu pruzili, int
# #e = np.full((1,2),5, dtype=float)  # sad bi ispisao 5.    
# print(e)
# f = np.eye(2)                       # jedinična matrica 2x2, na glavnoj dijagonali 1.
# print(f)


g = np.array([1, 2, 3], np.float32)
# duljina = len(g)
# print(duljina)
# h = g.tolist()                      # pretvra numpy listu u obicnu python listu
# print("ovo je polje h", h)

p = np.array([[10,20,30], [40,50,60]])
print(p)

novo_p = p.transpose()
print(novo_p)

# c = g.transpose()
# print("ovo je polje g", g)
polje=np.concatenate((a, g))
print(polje)

matrica = np.concatenate((p,b), axis=1) 
print(matrica)
