import argparse
import optuna
import yaml
import sys
import os

# Add src to path so we can import train_model
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from demec.training.train import train_model, load_config

def objective(trial, base_config):
    # Create a copy of the config to avoid modifying the original
    # (Though simple dict copy might not be deep enough if nested dicts are modified deeply, 
    # but here we modify top-level keys mostly. Let's do a somewhat safer copy)
    import copy
    config = copy.deepcopy(base_config)
    
    # --- Define Search Space ---
    
    # 1. Training Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    config['training']['lr'] = lr
    config['training']['batch_size'] = batch_size
    
    # 2. Model Hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    
    config['model']['hidden_dim'] = hidden_dim
    config['model']['dropout'] = dropout
    config['model']['num_layers'] = num_layers
    
    # If using GAT, tune heads
    if config['model']['architecture'] == 'gat':
        heads = trial.suggest_categorical("heads", [1, 2, 4, 8])
        config['model']['heads'] = heads

    # Print trial params
    print(f"\nTrial {trial.number}: lr={lr:.5f}, batch={batch_size}, hidden={hidden_dim}, layers={num_layers}, drop={dropout:.2f}")

    # --- Run Training ---
    try:
        # We might want to reduce epochs for tuning speed, or use the full amount.
        # Let's trust the config for epochs, or override if needed.
        # config['training']['epochs'] = 10 
        
        val_loss = train_model(config)
        return val_loss
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return a high loss to prune this trial
        return float('inf')

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning with Optuna")
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config file")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--storage", type=str, default="sqlite:///db.sqlite3", help="Database URL for Optuna dashboard")
    parser.add_argument("--study_name", type=str, default="demec_optimization", help="Name of the study")
    
    args = parser.parse_args()
    
    base_config = load_config(args.config)
    print(f"Loaded base configuration from {args.config}")
    
    # Create study
    study = optuna.create_study(
        direction="minimize", 
        storage=args.storage, 
        study_name=args.study_name,
        load_if_exists=True
    )
    
    print(f"Starting optimization with {args.n_trials} trials...")
    
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)
    
    print("\noptimization finished!")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optional: Save best config
    best_config = base_config.copy()
    # Update with best params... (logic similar to objective function)
    # This is left as an exercise or manual step for now.

if __name__ == "__main__":
    main()
