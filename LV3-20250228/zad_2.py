import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("mtcars.csv")

# 1. Pomoću barplot-a prikažite na istoj slici potrošnju automobila s 4, 6 i 8 cilindara.
avg_mpg = data.groupby("cyl")["mpg"].mean()

plt.figure()
plt.bar(avg_mpg.index, avg_mpg.values, color=["green", "blue", "red"])
plt.title("1. Prosjecna potrosnja automobila po broju cilindara")
plt.xlabel("Broj cilindara")
plt.ylabel("Potrosnja (mpg)")
plt.show()

# 2. Pomoću boxplot-a prikažite na istoj slici distribuciju težine automobila s 4, 6 i 8 cilindara.
weights = [data[data.cyl == i]["wt"] for i in [4, 6, 8]]

plt.figure()
plt.boxplot(weights, labels=["4", "6", "8"])
plt.title("2. Distribucija tezine automobila po broju cilindara")
plt.ylabel("Masa")
plt.xlabel("Broj cilindara")
plt.show()

# 3. Pomoću odgovarajućeg grafa pokušajte odgovoriti na pitanje imaju li automobili 
# s ručnim mjenjačem veću potrošnju od automobila s automatskim mjenjačem? 

data["mjenjac"] = data["am"].map({0: "Automatski", 1: "Rucni"})

avg_mpg = data.groupby("mjenjac")["mpg"].mean()

plt.figure()
plt.bar(avg_mpg.index, avg_mpg.values, color=["blue", "green"])
plt.title("3. Ovisnost potrosnje o tipu mjenjaca")
plt.xlabel("Mjenjac")
plt.ylabel("Potrosnja (mpg)")
plt.show()

print("Automobili sa rucnim mjenjacem manje trose jer pruzaju optimalnije prilagodjavanje brzine")

# 4. Prikažite  na  istoj  slici  odnos  ubrzanja  i  snage  automobila  za  automobile  
# s  ručnim  odnosno  automatskim mjenjačem

auto = data[data["am"] == 0]
manual = data[data["am"] == 1]

plt.figure()
plt.scatter(auto["hp"], auto["qsec"], color="blue", label="Automatski mjenjac")
plt.scatter(manual["hp"], manual["qsec"], color="green", label="Ručni mjenjac")

plt.title("4. Odnos ubrzanja i snage automobila ovisno o mjenjacu")
plt.xlabel("Snaga (hp)")
plt.ylabel("Ubrzanje (qsec)")
plt.legend()

plt.show()
