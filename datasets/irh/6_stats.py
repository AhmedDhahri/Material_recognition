import pandas as pd


data = pd.read_csv("dataset.csv")


print(data["mat"].value_counts())
print("Total: ", data.shape[0])
