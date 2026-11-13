# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:26:17 2026
"""
import pandas as pd
from pathlib import Path
import optuna
import subprocess
import csv

project_dir = Path(__file__).resolve().parent
csv_path = project_dir / "atom_jepa_trials.csv"

start_trails = 8

# Setup study
sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=start_trails)

study = optuna.create_study(direction="minimize", study_name="optuna_atom_jepa_reg", load_if_exists=True, sampler=sampler)

# Create csv file for saving hyperparams
if not csv_path.exists():
    with csv_path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["trial_id", "encoder_lr", "readout_lr", "batch_size", "score"])

# Count trails already created
trials = pd.read_csv(csv_path)
trials_nr = len(trials)

with csv_path.open("a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    
    for _ in range(start_trails):
        
        # Setup new trail and get hyperparameter suggestions
        trial = study.ask()
        
        encoder_lr = trial.suggest_float("encoder_lr", 1e-6, 1e-4, log=True)
        readout_lr = trial.suggest_float("readout_lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        
        if trial.number < trials_nr:
            continue
        
        hyper_params = (f"train.epochs=15 " f"model.encoder_lr={encoder_lr} " f"model.readout_lr={readout_lr} " f"train.batch_size={batch_size} " f"hydra.run.dir=runs/optuna_atom_jepa/trial_{trial.number}")


        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        command = f'bsub -env "MODEL=atom_jepa,TASK=regression,OVERRIDES={hyper_params}" < finetune.sh'
        print(command)

        writer.writerow([trial.number, encoder_lr, readout_lr, batch_size, ""])




