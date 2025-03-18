import numpy as np

a = np.array([3,1,5], float)
b = np.array([2,4,8], float)
print(a+b)                          # bit ce a[0]+b[0] itd -> [ 5.  5. 13.]
print(a-b)                          # [ 1. -3. -3.]
print(a*b)                          # [ 6.  4. 40.]
print(a/b)                          # [1.5   0.25  0.625]

print(f"najmanji element:\t\t{a.min()}")                      # vraca najmanji element
print(f"indeks najmanjeg elementa:\t{a.argmin()}")            # vraca indeks najmanjeg elementa -> 1
print(f"najveci element:\t\t{a.max()}")                       # vraca najveci element
print(f"indeks najveceg elementa:\t{a.argmax()}")             # vraca indeks najveceg elementa -> 2
print(f"suma polja a:\t\t\t{a.sum()}")                        # sumira polje
print(f"srednja vrijednost:\t\t{a.mean()}")                   # vraca srednju vrijednost polja

print(np.mean(a))
print(np.max(a))
print(np.sum(a))

a.sort()
print(a)