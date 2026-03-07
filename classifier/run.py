import sys
import os

# Asegurar que Python pueda encontrar los módulos
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# Importar los scripts
from train.XGB.XGBoost_30_train import main as run_xgb
from train.SVM.SVM_30_train import main as run_svm
from train.NN.NN_30_train import main as run_nn
from train.Gluon.Gluon_30_train import main as run_gluon
from train.Gluon.extract_best_gluon import main as run_extract


def main():
    #cantidad de modleos a entrenar
    train_n=6

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