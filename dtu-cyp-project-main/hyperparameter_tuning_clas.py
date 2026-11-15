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
csv_path = project_dir / "atom_jepa_trials_clas.csv"

start_trails = 8

# Setup study
sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=start_trails)

study = optuna.create_study(direction="maximize", study_name="optuna_atom_jepa_clas", load_if_exists=True, sampler=sampler)

def setup_random_trials(trials):

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
            
            hyper_params = (f"train.epochs=15 " f"model.encoder_lr={encoder_lr} " f"model.readout_lr={readout_lr} " f"train.batch_size={batch_size} " f"hydra.run.dir=runs/optuna_atom_jepa_clas/trial_{trial.number}")


            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            command = f'bsub -env "MODEL=atom_jepa,TASK=classification,OVERRIDES={hyper_params}" < finetune.sh'
            print(command)

            writer.writerow([trial.number, encoder_lr, readout_lr, batch_size, ""])


def suggest_next_trial(trials):
    
    # Load in random and prior trials
    for row in trials.itertuples(index=False):
        saved_settings = {"encoder_lr": float(row.encoder_lr), "readout_lr": float(row.readout_lr), "batch_size": int(row.batch_size),}

        study.enqueue_trial(saved_settings)
        saved_trial = study.ask()

        saved_trial.suggest_float("encoder_lr", 1e-6, 1e-4, log=True)
        saved_trial.suggest_float("readout_lr", 1e-4, 1e-2, log=True)
        saved_trial.suggest_categorical("batch_size", [8, 16])

        study.tell(saved_trial, float(row.score))
    
    printable_study = study.trials_dataframe(attrs=("number", "value", "params")).to_string
    print(printable_study)
    
    
    #Suggest next params
    next_trial = study.ask()

    encoder_lr = next_trial.suggest_float("encoder_lr", 1e-6, 1e-4, log=True)
    readout_lr = next_trial.suggest_float("readout_lr", 1e-4, 1e-2, log=True)
    batch_size = next_trial.suggest_categorical("batch_size", [8, 16])
    
    #print(f"Suggested settings for trial {next_trial.number}")
    #print(next_trial.params)
    
    hyper_params = (f"train.epochs=15 " f"model.encoder_lr={encoder_lr} " f"model.readout_lr={readout_lr} " f"train.batch_size={batch_size} " f"hydra.run.dir=runs/optuna_atom_jepa_clas/trial_{next_trial.number}")


    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    command = f'bsub -env "MODEL=atom_jepa,TASK=classification,OVERRIDES={hyper_params}" < finetune.sh'
    print(command)

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([next_trial.number, encoder_lr, readout_lr, batch_size, ""])
    
    return
    
if __name__ == "__main__":
    # Create csv
    if not csv_path.exists():
        with csv_path.open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["trial_id", "encoder_lr", "readout_lr", "batch_size", "score"])

    trials = pd.read_csv(csv_path)

    if len(trials) < start_trails:
        setup_random_trials(trials)
    else:
        suggest_next_trial(trials)



