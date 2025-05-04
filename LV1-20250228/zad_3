brojevi = []

while True:
    unos = input("Unesite broj ili ""done"" ako ste zavrsili sa unosom: ")
    
    if unos.lower() == 'done':
        break

    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Greska. Unesite broj ili ""done"" za kraj.")


print(f"Broj unosa: {len(brojevi)}")
print(f"Srednja vrijednost: {sum(brojevi) / len(brojevi):.2f}")
print(f"Minimalna vrijednost: {min(brojevi)}")
print(f"Maksimalna vrijednost: {max(brojevi)}")

brojevi.sort()
print(f"Sortirana lista: {brojevi}")