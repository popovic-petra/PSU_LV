# Napišite Python skriptu koja će učitati tekstualnu datoteku naziva SMSSpamCollection.txt. 
# Ova datoteka sadrži 425 SMS poruka pri čemu su neke označene kao spam, a neke kao ham. Primjer dijela datoteke:
# ham Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
# ham Ok lar... Joking wif u oni...
# spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken's stuff!
# ham Yup next stop.
# a) Izračunajte koliki je prosječan broj riječi u SMS porukama koje su tipa ham, a koliko je prosječan broj riječi u porukama koje su tipa spam.
# b) Koliko SMS poruka koje su tipa spam završava uskličnikom ?

with open('SMSSpamCollection.txt', 'r') as file:
    lines = file.readlines()

ham_br, spam_br, ham_rijeci, spam_rijeci, spam_usklicnik = 0,0,0,0,0

for line in lines:
    label, message = line.split('\t', 1)    # pomocu taba dijelimo poruku na dva dijela, oznaku i poruku
    words = message.split()                 # dijelimo rijeci poruke koje kasnije brojimo
    
    num_words = len(words)
    
    if label == 'ham':
        ham_br += 1
        ham_rijeci += num_words
    elif label == 'spam':
        spam_br += 1
        spam_rijeci += num_words
        
        if message.strip().endswith('!'):
            spam_usklicnik += 1

avg_ham = ham_rijeci / ham_br if ham_br > 0 else 0
avg_spam = spam_rijeci / spam_br if spam_br > 0 else 0

print(f"Broj rijeci u porukama tipa ham: {avg_ham}")
print(f"Broj rijeci u porukama tipa spam: {avg_spam}")
print(f"Broj spam poruka koje završavaju uskličnikom: {spam_usklicnik}")

