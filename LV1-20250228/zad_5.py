# skripta koja ucitava txt datoteku song.txt
# potrebno je napraviti rjecnik koji kao 
# kljuceve koristi sve razlicite rijeci koje se pojavljuju u datoteci
# dok su vrijednosti jednake broju puta koliko se svaka rijec (kljuc) pojavljuje u datoteci
# koliko rijeci se pojavljuje samo jednom ? ispisite ih

from collections import Counter

with open("song.txt") as file:
    rijec = file.read().lower().split()

rijecnik = dict(Counter(rijec))

# Pronalaženje riječi koje se pojavljuju samo jednom
jedinstvene_rijeci = []
for word, count in rijecnik.items():
    if count == 1:
        jedinstvene_rijeci.append(word)


print(f"Rijeci koje se pojavljuju samo jednom: {len(jedinstvene_rijeci)}")
print(jedinstvene_rijeci)

print("Ostale rijeci:")
print(rijecnik)