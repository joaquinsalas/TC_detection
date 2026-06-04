import sys
import os
import warnings

# Asegurar que Python pueda encontrar los módulos
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, root_dir)

# Importar los scripts
from classifier.train.XGB.XGBoost_30_train import main as run_xgb
from classifier.train.SVM.SVM_30_train import main as run_svm
from classifier.train.NN.NN_30_train import main as run_nn
from classifier.train.Gluon.Gluon_30_train import main as run_gluon
from classifier.train.Gluon.extract_best_gluon import main as run_extract


def main():
    #cantidad de modleos a entrenar
    train_n=30

    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("\n==============================")
    print("Entrenando clasificadores XGBoost")
    print("==============================\n")
    run_xgb(train_n)

    print("\n==============================")
    print("Entrenando clasificadores SVM")
    print("==============================\n")
    run_svm(train_n)

    print("\n==============================")
    print("Entrenando clasificadores NN")
    print("==============================\n")
    run_nn(train_n)

    print("\n==============================")
    print("Entrenando ensamblador AutoGluon")
    print("==============================\n")
    run_gluon(train_n)

    print("\n==============================")
    print("ENTRENAMIENTO COMPLETO, EMPIEZA EXTRACCIÓN DLE MEJOR ENSAMBLE")
    print("==============================\n")
    run_extract(train_n)


if __name__ == "__main__":
    main()
