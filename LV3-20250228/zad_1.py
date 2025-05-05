import pandas as pd

data = pd.read_csv('mtcars.csv')
#print(data)

# 1. Kojih 5 automobila ima najveću potrošnju (manji mpg)? (koristite funkciju sort)
data_greates_mpg = data.iloc[:,[0,1]]
print(f"1. 5 automobila s najvecom potrosnjom:\n{data_greates_mpg.sort_values(by='mpg').head(5)}")

# 2. Koja tri automobila s 8 cilindara imaju najmanju potrošnju (veci mpg)?
data_8cyl = data[(data.cyl==8)].iloc[:,[0,1,2]]
print(f"2. 3 automobila s 8 cilindara i najmanjom potrosnjom:\n{data_8cyl.sort_values(by='mpg').tail(3)}")

# 3. Kolika je srednja potrošnja automobila sa 6 cilindara? 
data_6cyl = data[(data.cyl==6)]
print(f"3. Srednja potrosnja automobila sa 6 cilindara\n{data_6cyl['mpg'].mean()}")

# 4. Kolika je srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs?
data_4cyl = data[(data.cyl==4) & (data.wt < 2.2) & (data.wt > 2.0)]
print(f"4. Srednja potrosnja automobila sa 4 cilindra mase između 2000 i 2200 lbs\n{data_4cyl['mpg'].mean()}")

# 5. Koliko je automobila s ručnim, a koliko s automatskim mjenjačem u ovom skupu podataka? 
# am - 0 -> automatski; 1 -> rucni
num_automatic = len(data[data['am'] == 0])
num_manual = len(data[data['am'] == 1])

print(f"5. Broj automobila s automatskim mjenjacem: {num_automatic}\nBroj automobila s rucnim mjenjacem: {num_manual}")

# 6. Koliko je automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga? 
data_hp = data[(data['am'] == 0) & (data.hp > 100)]
# print(len(data_hp))
# print(data_hp[['car', 'am', 'hp']])
print(f"6. Broj automobila s automatskim mjenjacem i snagom preko 100 konjskih snaga je: {(len(data_hp))}")

# 7. Kolika je masa svakog automobila u kilogramima?
data['masa_kg'] = data['wt'] * 1000 * 0.453592
print(f"7. Masa svakog automobila u kilogramima:\n{data[['car', 'masa_kg']]}")

