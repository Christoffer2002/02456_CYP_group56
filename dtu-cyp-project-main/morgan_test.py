from pathlib import Path
import pandas as pd
import numpy as np

# Directory setup
root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data/features"
#print("Test \n", data_dir, "\n Test")

file = np.load(data_dir / "morgan_3e44eceb9f.npz")

#keys = np.lib.npyio.NpzFile.keys(file)

#print(keys)