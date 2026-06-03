import sys
import os
import warnings


# Asegurar que Python pueda encontrar los módulos
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# Importar los scripts
from best_tcs.best_tcs import main as run_dates_and_names
from umbrales import main as intensive_GridSearch
from get30_umbrales_funciona import main as clean_db
from show_3d_db import main as graph_db


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    print("\n==============================")
    print("Generando la lista de nombres y de fechas a estudiar")
    print("==============================\n")
    run_dates_and_names()

    """print("\n==============================")
    print("Caracteriza clusteres usando busqueda estilo GridSearch")
    print("==============================\n")
    intensive_GridSearch()

    print("\n==============================")
    print("Limpia la base de datos ")
    print("==============================\n")
    clean_db()
    """
    """print("\n==============================")
    print("Visualiza base de datos ")
    print("==============================\n")
    color = 'white'
    graph_db()"""




if __name__ == "__main__":
    main()