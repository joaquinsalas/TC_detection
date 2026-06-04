import os
import json
import shutil

def load_base_model_seeds(json_path):
    with open(json_path, "r") as f:
        seeds_dict = json.load(f)
    NN_seeds  = seeds_dict["NN"]
    XGB_seeds = seeds_dict["XGB"]
    SVM_seeds = seeds_dict["SVM"]
    return NN_seeds, XGB_seeds, SVM_seeds

class ClasificadorHurakan:
    #NN_seeds=[17,3,11] #modelos 0, 1, 2
    #XGB_seeds=[0,4,19] #modelos 3, 4, 5
    #SVM_seeds=[4,28,26] #modelos 6, 7, 8
    names = ['NN', 'XGB', 'SVM']
    def __init__(self, NN_seeds, XGB_seeds, SVM_seeds):
        self.NN_seeds = NN_seeds
        self.XGB_seeds = XGB_seeds
        self.SVM_seeds = SVM_seeds
        self.models_seeds = [NN_seeds, XGB_seeds, SVM_seeds]

    def nn_bundle_predict_proba(self, nn_bundle, X_test):
        import torch
        from classifier.train.NN.NN_30_train import FlexibleNN

        """nn_bundle: dict con keys 'scaler', 'model_state', 'params'."""
        scaler = nn_bundle["scaler"]
        state_dict = nn_bundle["model_state"]
        params = nn_bundle["params"]

        # Asegura numpy 2D
        X_np = X_test.values if hasattr(X_test, "values") else X_test
        Xs = scaler.transform(X_np)  # usa el scaler guardado en el bundle

        # Reconstruir la arquitectura exacta
        input_dim = Xs.shape[1]
        n_layers = params.get("n_layers", 1)
        hidden_dims = [params.get(f"n_units_layer_{i}", 64) for i in range(n_layers)]
        dropout = params.get("dropout", 0.0)
        activation_name = params.get("activation", "relu")

        model = FlexibleNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, activation_name=activation_name)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        with torch.no_grad():
            logits = model(torch.tensor(Xs, dtype=torch.float32)).squeeze(1)
            probas = torch.sigmoid(logits).cpu().numpy()
        return probas

    def svm_bundle_predict_proba(self, svm_bundle, X_test):
        scaler = svm_bundle["scaler"]
        model = svm_bundle["model"]

        # Asegura numpy 2D
        X_np = X_test.values if hasattr(X_test, "values") else X_test
        Xs = scaler.transform(X_np)
        probas = model.predict_proba(Xs)[:, 1]
        return probas

    # --- Helper para cargar modelos y obtener probas ---
    def predict_with_model(self, model_info, X):
        # Si es un bundle NN (dict), reconstruimos; si no, asumimos sklearn (XGB Pipeline)
        if isinstance(model_info, dict) and "model_state" in model_info and "scaler" in model_info:
            probas = self.nn_bundle_predict_proba(model_info, X)
        # Si es un bundle SVM
        elif isinstance(model_info, dict) and "model" in model_info and "scaler" in model_info:
            probas = self.svm_bundle_predict_proba(model_info, X)
        # sirve para XGB
        elif "sklearn.pipeline" in str(type(model_info)):
            # predict_proba(X) ya incluye el escalado
            probas = model_info.predict_proba(X)[:, 1]
        else:
            print('tipo de modelo no reconocido')
            probas = None
        return probas

    def preprocess_input(self, df, models_dir):
        import pickle
        import pandas as pd

        files = []
        #usar los clasificadores NN, XGB y SVM para generar el dataset stackeado
        for model_seed_list, name in zip(self.models_seeds, self.names):
            for seed in model_seed_list:
                #cargar el modelo correspondiente
                file = os.path.join(models_dir, f"{name}_classifier_seed_{seed}.pkl")
                files.append(file)

        # 3. Calcula features stacking: proba de cada modelo base
        preds = pd.DataFrame()
        for idx, f in enumerate(files):
            with open(f, "rb") as pf:
                model_info = pickle.load(pf)
            col_name = f"model_{idx}"
            preds[col_name] = self.predict_with_model(model_info, df)

        return preds


    def clasificar(self, df, ensemble_dir, models_dir):
        from autogluon.tabular import TabularPredictor

        #primero obtiene las probabilidades de cada clasificador base
        preds = self.preprocess_input(df, models_dir)

        #cargar el modelo desde model_dir
        predictor = TabularPredictor.load(ensemble_dir)
        probabilidades = predictor.predict_proba(preds)
        resultado_final = predictor.predict(preds)
        return probabilidades, resultado_final
    

#borra carpetas, si existen, y las vuleve a crear vacías
def reset_output_paths(dirs=None, files=None):
    if dirs is None:
        dirs = []
    if files is None:
        files = []

    # Resetear carpetas
    for directory in dirs:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    # Borrar archivos si existen
    for file_path in files:
        if os.path.isfile(file_path):
            os.remove(file_path)
