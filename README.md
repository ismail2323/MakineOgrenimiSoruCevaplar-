# MakineOgrenimiSoruCevaplar-
# Soru 1 için
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = sklearn.datasets.load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head())
print(df.isnull().sum())
print(df.describe())
df.hist(figsize=(15,15))
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap="RdBu")
plt.show()

# Soru 2 için 
import random
keys = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
values = [random.randint(0,100) for _ in range(10)]
my_dict = dict(zip(keys, values))
print(my_dict)
del my_dict["a"]
print(my_dict)
my_dict["k"] = 50
print(my_dict)
my_dict["b"] = 75
print(my_dict)

# Soru 3 için
my_tuple = tuple(random.randint(0,100) for _ in range(7))
print(my_tuple)
my_list = [random.randint(0,100) for _ in range(7)]
print(my_list)
my_set = set(random.randint(0,100) for _ in range(7))
print(my_set)

# Soru 4 için
import numpy as np
import pandas as pd

data1 = np.random.randint(0,100, size=(20,7))
df1 = pd.DataFrame(data1, columns=list("ABCDEFG"))
print(df1)
data2 = np.random.randint(0,100, size=(30,7))
df2 = pd.DataFrame(data2, columns=list("ABCDEFG"))
print(df2)
df = pd.concat([df1, df2], ignore_index=True)
print(df)
df["H"] = df.iloc[:,0] + df.iloc[:,-1]
print(df)
