import numpy as np

# postavi seed generatora brojeva, moze se koristiti bilo koji cijeli broj, 
# za svaki broj drugaciji se brojevi genenriraju, ali isti za svako ponovno pokretanje programa

np.random.seed(56)              
rNumbers = np.random.rand(10)   # generiraj 10 slučajnih brojeva od 0.0 do 1.0
rNumbers = np.random.randint(1, 11, size=10)  # 10 brojeva u rasponu [1, 10]
print(rNumbers)
print(rNumbers.mean())