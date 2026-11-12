# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:31:30 2026

@author: chris
"""
import json
from pathlib import Path
import optuna
import subprocess
import sys

# Paths
project_dir = Path(__file__).resolve().parent
output = Path("runs/optuna_practice")

validation_metrics_run_1 = project_dir / output / "manual_trial/validation_metrics.json"

# Load file
metrics = json.loads(validation_metrics_run_1.read_text(encoding="utf-8"))

score = metrics["macro_ST-RAE"]
print(score)


study = optuna.create_study(direction="minimize")


study.enqueue_trial({"readout_lr": 0.0005})
trial = study.ask() # start trial
learning_rate = trial.suggest_float("readout_lr", 1e-4, 1e-3) # 
study.tell(trial, score)

print("Initial trail - manual learning rate:", learning_rate)


for i in range(25):
    
    next_trial = study.ask()
    next_lr = next_trial.suggest_float("readout_lr", 1e-4, 1e-3)
    
    command = [sys.executable, "finetune.py", "model=morgan_rdkit_mlp", "task=regression", "data=../data", "device=cpu", "train.epochs=1", f"model.readout_lr={next_lr}", f"hydra.run.dir=runs/optuna_practice/trial_{i}"]
    subprocess.run(command, cwd=project_dir, check=True)
    
    metrics_path = project_dir / output / f"trial_{i}" / "validation_metrics.json"
    trial_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    
    trial_score = trial_metrics["macro_ST-RAE"]
    study.tell(next_trial, trial_score)
    print(f"Trial {i}: lr={next_lr:.6g}, score={trial_score:.4f}")
    

print(study.best_params)
print(study.best_value)
