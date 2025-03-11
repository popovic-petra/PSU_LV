# program zahtijeva unos brojeca u beskonacnoj petlji sve dok korisnike ne upise Done
# brojevi se spremaju u listu
# potrebno ispisati koliko je brojeva korisnik unio, te srednju, minimalnu i maksimalnu vrijednost
# sortirajte listu i ispisite na ekran
# osigurati poruku greske uslijed krivog unosa

lista = []
brojac = 0

while True:
    unos = input("Unesite broj: ")
    if unos.lower()=="done":
        break

    try:
        broj = int(unos)
        lista.append(broj)
        brojac+=1
    except ValueError:
        print("Pogresan unos")

#print(lista)
#print(len(lista))
#umjesto brojac moglo je i len(lista)
print(f"Uneseno brojeva: {brojac}")
print(f"Maksimalna vrijednost: {max(lista)}")
print(f"Minimalna vrijednost: {min(lista)}")
print(f"Srednja vrijednost: {sum(lista)/brojac}")
#sortirana_lista = lista.sort() #sortira listu i vraca none, stoga...

lista.sort()
print(f"Sortirana lista: {lista}")