import sys
import os
import warnings

# Asegurar que Python pueda encontrar los módulos
# subir 1 niveles: database_creation → TC_detection
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, root_dir)

def main():
    from database_creation.best_tcs.best_tcs import main as run_dates_and_names
    from database_creation.show_3d_db import main as graph_db
    from database_creation.umbrales import main as intensive_GridSearch
    from database_creation.get30_umbrales_funciona import main as clean_db

    color_graph = 'white'
    warnings.filterwarnings("ignore", category=UserWarning)

    print("\n==============================")
    print("Generando la lista de nombres y de fechas a estudiar")
    print("==============================\n")
    run_dates_and_names()

    print("\n==============================")
    print("Caracteriza clusteres usando busqueda estilo GridSearch")
    print("==============================\n")
    intensive_GridSearch(color_graph)

    print("\n==============================")
    print("Limpia la base de datos ")
    print("==============================\n")
    clean_db()
    
    print("\n==============================")
    print("Visualiza base de datos ")
    print("==============================\n")
    graph_db()




if __name__ == "__main__":
    main()
