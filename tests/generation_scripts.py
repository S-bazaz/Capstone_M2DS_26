import matplotlib.pyplot as plt
import numpy as np
from graph_class import *
from pathlib import Path



def generate_and_save_graphs(node_steps, segment_steps, n_variants=10, x_size=10, y_size=10):
    # 1. Définition des lois de probabilité
    distributions = {
        "uniform": lambda x: 1,
        "inverse": lambda x: 1/(1e-6 + x),
        "sqrt_inv": lambda x: 1/np.sqrt(1e-6 + x),
        "affine": lambda x: max(0, (1 - x/np.sqrt(x_size**2 + y_size**2)))
    }

    # 2. Racine du dossier de tests
    base_dir = Path("tests/plots_reseau")
    
    for dist_name, dist_func in distributions.items():
        # Niveau 1 : Distribution
        dist_dir = base_dir / dist_name
        
        for n_n in node_steps:
            # Niveau 2 : Nombre de noeuds
            node_dir = dist_dir / f"nodes_{n_n}"
            
            for n_s in segment_steps:
                # Niveau 3 : Nombre de segments
                segment_dir = node_dir / f"segments_{n_s}"
                segment_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"Génération : {dist_name} | {n_n} nodes | {n_s} segments...")

                for v in range(1, n_variants + 1):
                    # Initialisation (Assure-toi que pick_point_dist est utilisé dans ton generate_pipeline)
                    graph = DynamicGraph(size_x=x_size, size_y=y_size, n_node=n_n, n_segments=n_s, f=dist_func)
                    
                    plt.figure(figsize=(10, 10))
                    
                    # Dessin des segments
                    for s in graph.segment_list.l_segments:
                        plt.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y], color='black', alpha=0.6, linewidth=1.2)
                    
                    # Dessin des points
                    xs = [p.x for p in graph.point_list.l_points]
                    ys = [p.y for p in graph.point_list.l_points]
                    plt.scatter(xs, ys, c='red', s=20, zorder=3)
                    
                    plt.title(f"Dist: {dist_name} | N: {n_n} | S: {n_s} | Var: {v}")
                    plt.axis('equal')
                    plt.grid(True, linestyle=':', alpha=0.4)

                    # Sauvegarde dans l'arborescence
                    filepath = segment_dir / f"exemple_{v}.png"
                    plt.savefig(filepath, bbox_inches='tight')
                    plt.close()



