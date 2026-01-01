import pandas as pd

df = pd.read_csv("ropcsv.csv")

df["Source"] = df["Source"].str.replace(
    r"^.*ROP\*Lot\*2",
    lambda _: r"C:\Users\HP\Desktop\ROPDetectionAIML\data\images",
    regex=True
)

df["Source"] = df["Source"].str.replace("*", " ", regex=False)

df.to_csv("ropfinal.csv", index=False)
