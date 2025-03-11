# program koji zahtijeva unos imena tekstualne datoteke
# program treba traziti linije X-DSPAM-Confidence: <neki_broj>
# predstavljaju pouzdanost koristenog spam filtra
# potrebno je izracunati srednju vrijednost pouzdanosti
# koristi datoteke mbox.txt i mbox-short.txt

def srednja_vrijednost(filename):
    try:
        fhand = open(filename, "r")  
        values = []

        for line in fhand:
            if line.startswith("X-DSPAM-Confidence:"):
                values.append(float(line.split(":")[1]))

        fhand.close()  

        if values:
            print(f"Average X-DSPAM-Confidence: {sum(values) / len(values)}")
        else:
            print("Nema pronađenih vrijednosti za X-DSPAM-Confidence.")
    
    except FileNotFoundError:
        print(f"Greška: Datoteka '{filename}' ne postoji.")

srednja_vrijednost("mbox.txt")
srednja_vrijednost("mbox-short.txt")