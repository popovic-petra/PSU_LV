# program koji zahtijeva od korisnika unos radnih sati te koliko je placen po satu
# koristiti metodu input()
# izracunajte koliko je korisnik zaradio i ispisite
# prepravite rjesenje da ukupni iznos izracunavate u posebnoj funkc total_euro

sati = int(input("Unesite broj radnih sati: "))
satnica = float(input("Unesite satnicu: "))

def total_euro(sati, satnica):
    return float(sati*satnica)

#print("Za", sati, "h, uz satnicu", satnica, "eura, zaradite", total_euro(sati, satnica), "eura")
print(f"Radni sati: {sati}h\neura/h: {satnica}\nUkupno: {total_euro(sati, satnica)} eura")