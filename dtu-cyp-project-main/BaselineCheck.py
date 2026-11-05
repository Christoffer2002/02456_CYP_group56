# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:19:32 2026

@author: 45401
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# run_dir = Path(r"C:\Users\45401\OneDrive\Dokumenter\GitHub\02456_CYP_group56\dtu-cyp-project-main\runs\classification\morgan_rdkit_lr_0.01_hidden1024")
run_dir = Path(r"C:\Users\45401\OneDrive\Dokumenter\GitHub\02456_CYP_group56\dtu-cyp-project-main\runs\classification\morgan_rdkit_lr_0.0001")

history = pd.read_json(run_dir / "history.jsonl", lines=True)

plt.figure()
plt.plot(history["epoch"], history["train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.grid()
plt.show()

plt.figure()
plt.plot(history["epoch"], history["CYP2D6_is_TDI"].apply(lambda scores: scores["MCC"]), label="CYP2D6")
plt.plot(history["epoch"], history["CYP3A4_is_TDI"].apply(lambda scores: scores["MCC"]), label="CYP3A4")
plt.xlabel("Epoch")
plt.ylabel("Validation MCC")
plt.legend()
plt.grid()
plt.show()