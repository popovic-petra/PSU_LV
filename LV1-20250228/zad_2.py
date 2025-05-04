try:
    broj = float(input("Unesite ocjenu izmedju 0.0 i 1.0: "))

    if not (0.0 <= broj <= 1.0):
        raise ValueError("Broj mora biti izmedju 0.0 i 1.0!")
    
    if broj >= 0.9:
        print("A")
    elif broj >= 0.8:
        print("B")
    elif broj >= 0.7:
        print("C")
    elif broj >= 0.6:
        print("D")
    else:
        print("F")

except ValueError as e:
    print("Greska:", e)
