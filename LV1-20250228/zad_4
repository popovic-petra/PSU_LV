# mbox.txt
# mbox-short.txt

ime_datoteke = input("Unesite ime datoteke: ")
try:
    with open(ime_datoteke, 'r') as datoteka:
        ukupno = 0
        broj_linija = 0

        for linija in datoteka:
            linija = linija.strip()
            if linija.startswith("X-DSPAM-Confidence:"):
                try:
                    broj = float(linija.split(":")[1].strip())
                    ukupno += broj
                    broj_linija += 1
                except ValueError:
                    print("Greska u liniji", linija)

        if broj_linija == 0:
            print("Nema pronadjenih podataka")
        else:
            prosjek = ukupno / broj_linija
            print(f"Average X-DSPAM-Confidence: {prosjek}")

except FileNotFoundError:
    print("Greska: Datoteka nije pronadjena.")
