# este es el mero mero para NN
#falta comentarlo en inglés

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import optuna
from sklearn.model_selection import StratifiedKFold
from train_common import split_and_preprocess, calcula_PR_ascendente, count_dir

# --- Red neuronal en PyTorch ---
class FlexibleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation_name):
        super().__init__()
        layers = []
        prev_dim = input_dim
        # Elige activación según nombre
        if activation_name == "relu":
            activation = nn.ReLU()
        elif activation_name == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # output
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def entrenar_modelo(X, y, X_val, y_val, hidden_dims, dropout, activation_name, lr, batch_size, max_epochs, anneal_strategy, weight_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlexibleNN(X.shape[1], hidden_dims, dropout, activation_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).reshape(-1,1))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*5, steps_per_epoch=steps_per_epoch, epochs=max_epochs, anneal_strategy=anneal_strategy
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(np.array(y_val), dtype=torch.float32).reshape(-1,1).to(device)
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss.item()
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
        if wait > 10: #early stopping
            break
    model.load_state_dict(best_state)
    return model, X_val_t


# -------------- Optuna Objective -------------
def objective(trial, X, y,n_folds):
    n_layers = trial.suggest_int("n_layers", 1, 2)
    hidden_dims = [trial.suggest_int(f"n_units_layer_{i}", 8, 32) for i in range(n_layers)]
    activation_name = trial.suggest_categorical("activation", ["relu", "leakyrelu", "tanh"])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    anneal_strategy = trial.suggest_categorical("anneal_strategy", ["cos", "linear"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    max_epochs = 200
    
    # Cross-validation
    auc_prs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model, X_val_t = entrenar_modelo(
            X_tr, y_tr, X_val, y_val, hidden_dims, dropout, activation_name,
            lr, batch_size, max_epochs, anneal_strategy, weight_decay
        )

        device = next(model.parameters()).device  # detecta si está en cuda o cpu
        with torch.no_grad():
            logits = model(torch.tensor(X_val_t, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        auc_pr, _, _ = calcula_PR_ascendente(y_val, probs)
        auc_prs.append(auc_pr)
    return np.mean(auc_prs)


# main ------------------------------------------------------------------------
def main(train_n):
    csv_path = "database_creation/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    out_dir = "classifier/modelos/NN"
    best_dir = 'classifier/train/NN/best'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    

    best_score_pr = -np.inf
    best_model = None
    best_scaler = None
    best_seed = None
    best_params = None
    best_params_best = None
    params_list = []
    n_folds = 5


    last_seed = count_dir(out_dir)
    seeds_list = list(range(last_seed, train_n))
    for seed in seeds_list:
        print(f"\n=== SEMILLA {seed} ===")
        X_train, X_test, y_train, y_test = split_and_preprocess(
            df, 'fecha_prediccion', '2023-01-01', '2024-12-31', 'label', seed
        )

        # Escalado (como siempre)
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train)
        X_test_np  = scaler.transform(X_test)

        # Split interno para validación (usada por Optuna)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_np, y_train, test_size=0.2, stratify=y_train, random_state=seed)

        X_arr = np.array(X_train_np)
        y_arr = np.array(y_train)
        # Optuna hyperparameter tuning
        def optuna_objective(trial):
            return objective(trial, X_arr, y_arr, n_folds)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(optuna_objective, n_trials=100, show_progress_bar=True)

        best_params = study.best_params
        print(f"Mejores parámetros Optuna:", best_params)
        # Añade info de la semilla
        params_row = best_params.copy()
        params_row['seed'] = seed
        params_list.append(params_row)
        
        # Entrenamiento final con mejores params y todo el train
        # Es un amadresota
        ################################
        n_layers = best_params['n_layers']
        hidden_dims = [best_params[f'n_units_layer_{i}'] for i in range(n_layers)]
        activation_name = best_params['activation']
        dropout = best_params['dropout']
        lr = best_params['lr']
        batch_size = best_params['batch_size']
        anneal_strategy = best_params['anneal_strategy']
        weight_decay = best_params['weight_decay']
        max_epochs = 200

        final_model, _ = entrenar_modelo(X_train_np, y_train, X_val, y_val, hidden_dims, dropout, activation_name, lr, batch_size, max_epochs, anneal_strategy, weight_decay)
        ####################################


        # Guardar modelo y scaler juntos
        with open(f"{out_dir}/NN_classifier_seed_{seed}.pkl", "wb") as f:
            pickle.dump({"scaler": scaler, "model_state": final_model.state_dict(), "params": best_params}, f)

        # Evaluación en test
        device = next(final_model.parameters()).device
        with torch.no_grad():
            logits = final_model(torch.tensor(X_test_np, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        auc_roc = roc_auc_score(y_test, probs)
        auc_pr, _, _ = calcula_PR_ascendente(y_test, probs)

        if auc_pr > best_score_pr:
            best_score_pr = auc_pr
            best_model = final_model
            best_scaler = scaler
            best_seed = seed
            best_params_best = best_params

        print(f"Seed={seed}: AUC-PR={auc_pr:.3f}  AUC-ROC={auc_roc:.3f}")

    # Guardar mejor modelo
    if best_params_best is not None:
        with open(os.path.join(best_dir, f'Best_NN_classifier_seed_{best_seed}.pkl'), "wb") as f:
            pickle.dump({"scaler": best_scaler, "model_state": best_model.state_dict(), "params": best_params_best}, f)
        print(f"\nMejor modelo NN-PyTorchOptuna: seed={best_seed} con AUC-PR={best_score_pr:.3f}")

        df_params = pd.DataFrame(params_list)
        df_params.to_csv(os.path.join(best_dir, 'best_params.csv'), index=False)

if __name__ == "__main__":
    main(train_n)