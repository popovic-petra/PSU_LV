rs = input("Unesite broj radnih sati: ")
rs = float(rs)

s = input("Unesite kolika je satnica: ")
s = float(s)

rezultat = s*rs

print(f"Radni sati: {rs} h\neura/h: {s}\nUkupno: {rezultat} eura")

def total_euro(rs, s):
    rezultat = rs*s
    return rezultat

print(f"Radni sati: {rs} h\neura/h: {s}\nUkupno: {total_euro(rs, s)} eura")