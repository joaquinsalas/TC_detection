# Este códidgo sirve para genera la gráfica que muestra los umbrales en 3D

import pandas as pd
import matplotlib.pyplot as plt




def main():

    plt.rcParams.update({
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    # Leer el archivo CSV
    path = 'database_creation/confirmed_umbrales_ciclones_old.csv'
    df = pd.read_csv(path, parse_dates = ['fecha_prediccion']) 
    # ------------------------------------------------------------------------------------
    labels = ['label'] 

    for label in labels: 
        # Crear la figura y el eje
        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white') 

        # Pintar cada “pane” (cara del cubo 3D) en negro
        # quitar el relleno de cada “pane” para que no se vea el gris
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # pintar la rejilla de un color claro
        ax.grid(True)
        ax.xaxis._axinfo["grid"]['color'] = "gray"
        ax.yaxis._axinfo["grid"]['color'] = "gray"
        ax.zaxis._axinfo["grid"]['color'] = "gray"

        # Graficar puntos con label 0 (rojo)
        df_negativos = df[df[label] == 0]
        ax.scatter(
            df_negativos["dispersión_km_best_cluster"],
            df_negativos["n_trayectorias_best_cluster"],
            df_negativos["horas_diff_estimadas"],
            color="red",
            alpha=0.4,   # <-- transparencia
            label=r"Non-tropical storm"
        )

        # Graficar puntos con label 1 (verde)
        df_positivos = df[df[label] == 1]
        ax.scatter(
            df_positivos["dispersión_km_best_cluster"],
            df_positivos["n_trayectorias_best_cluster"],
            df_positivos["horas_diff_estimadas"],
            color="green",
            alpha=0.4,   # <-- transparencia
            label=r"Tropical storm"
        )

        # Etiquetas y título
        ax.set_xlabel("Dispersion (km)") #"Dispersion (km)"
        ax.set_ylabel("Number of trajectories") #"Number of trajectories"
        ax.set_zlabel("Estimated lead time (hr)")  # "Hours before the cyclone"
        ax.set_title("Distribution of tropical and non-tropical storms based on dispersion, trajectories and estimated lead time")

        legend = ax.legend(
            facecolor="white",     # fondo blanco
            edgecolor="black",     # borde negro
            framealpha=1           # sin transparencia
        )
        # texto negro dentro de la leyenda
        for text in legend.get_texts():
            text.set_color("black")

        # Cuadrícula y mostrar
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"database_creation/figures/stats_{label}_3D.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.5 
        )
        plt.close(fig)

if __name__ == "__main__":
    main()