# este es el mero mero para NN
#

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import optuna
from sklearn.model_selection import StratifiedKFold

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

def preprocess(df, label, flag, seed = None):
    if flag =='train':
        # Balancear clases y normalizar X
        min_class = df[label].value_counts().min()
        df_0 = df[df[label] == 0].sample(min_class, random_state=seed)
        df_1 = df[df[label] == 1].sample(min_class, random_state=seed)
        df_bal = pd.concat([df_0, df_1]).sample(frac=1, random_state=seed)
    elif flag == 'test':
        df_bal = df

    # zona: pacifico=1, atlantico=0
    #df_bal['zona'] = np.where(df_bal['zona'] == 'pacifico', 1, 0)
    X = df_bal[["n_trayectorias_best_cluster", "dispersión_km_best_cluster",'horas_diff_estimadas']] #zona
    y = df_bal[label]
    return X, y
    

#divide los datos de entrenamiento (2023 y 2024) y test (2025) por fecha
def split_and_preprocess(df: pd.DataFrame, date_col: str,
                         train_start: str, train_end: str,
                         label: str, seed:int):
    df[date_col] = pd.to_datetime(df[date_col])

    # crear máscara
    start = pd.to_datetime(train_start)
    end   = pd.to_datetime(train_end)
    mask_train = df[date_col].between(start, end)

    # partir DataFrames
    df_train = df.loc[mask_train].reset_index(drop=True)
    df_test  = df.loc[~mask_train].reset_index(drop=True)
    X_train, y_train = preprocess(df_train, label, 'train', seed)
    X_test,  y_test  = preprocess(df_test,  label, 'test')
    return X_train, X_test, y_train, y_test

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


# Filtra los valores de Precision y Recall para que conforme el Recall dismiya, Precision aumente (y no decaiga)
def calcula_PR_ascendente(y_true, y_proba):
    precision0, recall0, _ = precision_recall_curve(y_true, y_proba)
    df_pr = pd.DataFrame({
        'recall':    recall0,
        'precision': precision0
    }).sort_values('recall', ascending=False).reset_index(drop=True)

    #Filtrar para que precision nunca baje
    prec_prev = 0.0
    mask = []
    for p in df_pr['precision']:
        if p >= prec_prev:
            mask.append(True)
            prec_prev = p
        else:
            mask.append(False)
    df_pr_f = df_pr[mask]

    recall_f = df_pr_f['recall'].values
    precision_f = df_pr_f['precision'].values

    #Calcular AUC-PR de la curva escalonada
    return auc(recall_f, precision_f), recall_f, precision_f

# main ------------------------------------------------------------------------
def main():
    csv_path = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/confirmed_umbrales_ciclones.csv"
    df = pd.read_csv(csv_path, parse_dates = ['fecha_prediccion'])

    out_dir = "/home/nathaliealvarez/Personal/umbral_definition/umbrales_Hurakan/classifier/modelos/NN"
    best_dir = '/home/nathaliealvarez/Personal/Repos/TC_detection/classifier/train/NN/best'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    

    best_score_pr = -np.inf
    best_model = None
    best_scaler = None
    best_seed = None
    best_params = None
    params_list = []
    n_folds = 5


    for seed in range(30):
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
    os.path.join(best_dir, f'Best_NN_classifier_seed_{best_seed}.pkl')
    with open(os.path.join(best_dir, f'Best_NN_classifier_seed_{best_seed}.pkl'), "wb") as f:
        pickle.dump({"scaler": best_scaler, "model_state": best_model.state_dict(), "params": best_params_best}, f)
    print(f"\nMejor modelo NN-PyTorchOptuna: seed={best_seed} con AUC-PR={best_score_pr:.3f}")

    df_params = pd.DataFrame(params_list)
    df_params.to_csv(os.path.join(best_dir, 'best_params.csv'), index=False)

if __name__ == "__main__":
    main()