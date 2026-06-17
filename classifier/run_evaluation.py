# En este archivo solo se hacen las evaluaciones y se generan métricas

import sys
import os
# Asegurar que Python pueda encontrar los módulos
# subir 1 niveles: classifier → TC_detection
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, root_dir)

import matplotlib.pyplot as plt
import warnings

# Importar los scripts
from classifier.evaluation.models_performance import main as run_models_performance
from classifier.evaluation.more_results import main as run_more_results
#from classifier.evaluation.evaluation_z_score import main as run_z_score
from classifier.evaluation.z_test import main as run_z_score



def main():
    
    color_graph = 'white'
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("\n==============================")
    print("Obteniendo métricas de desempeño")
    print("==============================\n")
    results = run_models_performance(color_graph)
    

    """("\n==============================")
    print("Obteniendo POD, FAR, Accuracy, Accuracy por hora, Matríz de confusión, y un db summary")
    print("==============================\n")
    #lead_time = -12
    lead_time = None
    run_more_results(color_graph, lead_time=lead_time)
    #del general, model=1 fue el mejor (en este caso se refiere a NN=20)"""

    """print("\n==============================")
    print("Calculando z_score de AUC-ROC Y AUC-PR con α = 0.05")
    print("==============================\n")
    run_z_score(results)
    """



if __name__ == "__main__":
    main()