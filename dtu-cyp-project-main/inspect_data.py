# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

# Directory setup
root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data"

# Load data
data = pd.read_csv(data_dir / "regression" / "train.csv")
classification = pd.read_csv(data_dir / "classification" / "train.csv")

#Shave off conf intervals
# data = data.iloc[:, [0, 1, 2, 3, 4, 5, 18, 19, 20]]

# Regression data
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Data shape & columns")
print(data.shape)
print(data.columns.tolist())

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Example of molecule 10:")
print(data.iloc[9])
#print(data["SMILES"].iloc[0])

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Number of NaNs")
print("CYP1A2_pIC50_direct_inhibition", data.iloc[:,2].isna().sum())
print("CYP2C9_pIC50_direct_inhibition", data.iloc[:,3].isna().sum())
print("CYP2D6_pIC50_direct_inhibition", data.iloc[:,4].isna().sum())
print("CYP3A4_pIC50_direct_inhibition", data.iloc[:,5].isna().sum())

# Statistics
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Count")
print("CYP1A2:", data.iloc[:, 2].count())
print("CYP2C9:", data.iloc[:, 3].count())
print("CYP2D6:", data.iloc[:, 4].count())
print("CYP3A4:", data.iloc[:, 5].count())
print("Mean")
print("CYP1A2:", data.iloc[:, 2].mean())
print("CYP2C9:", data.iloc[:, 3].mean())
print("CYP2D6:", data.iloc[:, 4].mean())
print("CYP3A4:", data.iloc[:, 5].mean())
print("std")
print("CYP1A2:", data.iloc[:, 2].std())
print("CYP2C9:", data.iloc[:, 3].std())
print("CYP2D6:", data.iloc[:, 4].std())
print("CYP3A4:", data.iloc[:, 5].std())

# Classification data
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Classification:")
print(classification.shape)
print("Example of molecule 1:")
print(classification.iloc[0])

#True and false count
CYP2D6_0 = classification.iloc[:, 2] == 0
CYP3A4_0 = classification.iloc[:, 3] == 0
CYP2D6_1 = classification.iloc[:, 2] == 1
CYP3A4_1 = classification.iloc[:, 3] == 1
print("CYP2D6:", CYP2D6_0.sum(), CYP2D6_1.sum())
print("CYP3A4:", CYP3A4_0.sum(), CYP3A4_1.sum())


