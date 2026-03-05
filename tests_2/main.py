from .objects.graph import RobustNetwork
from .solvers.naive_solver import NaiveSolver
from .solvers.solver_density_1 import DensitySolver
from .objects.evaluator import Evaluator 
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

def get_info_text(solver_name, metrics):
    """Génère la chaîne de caractères formatée pour l'affichage sur le graphique."""
    return (
        f"SOLVER: {solver_name.upper()}\n"
        f"{'-'*25}\n"
        f"Fixe (Supports)  : {metrics['weighted_fixed']:.2f}\n"
        f"Var. (Câbles)    : {metrics['weighted_variable']:.2f}\n"
        f"{'-'*25}\n"
        f"TOTAL SCORE      : {metrics['total_score']:.2f}"
    )

def print_compact_score(solver_name, solution, evaluator):
    """Affiche le score en console et retourne le dictionnaire des métriques."""
    metrics = evaluator.evaluate(solution)
    print(f"\n📊 {solver_name.upper()}")
    print(f"  > Support (fixe)   : {metrics['fixed_raw']} supports")
    print(f"  > Câble (variable)  : {metrics['variable_raw']:.2f}")
    print(f"  > TOTAL             : {metrics['total_score']:.2f}")
    print("-" * 30)
    return metrics

def plot_comparison(net, solutions_dict, save_path=None):
    """ Superpose les tracés des différents solveurs sur un seul graphe. """
    pos = nx.get_node_attributes(net.G, 'pos')
    plt.figure(figsize=(14, 10))
    
    # 1. Fond du réseau
    nx.draw_networkx_edges(net.G, pos, edge_color='lightgray', alpha=0.2, style=':')
    nx.draw_networkx_nodes(net.G, pos, node_size=5, node_color='lightgray', alpha=0.3)
    
    # 2. Dessin des routes par solveur
    colors = ['#1f77b4', '#ff7f0e'] # Bleu pour Naïf, Orange pour Density
    for (name, sol), color in zip(solutions_dict.items(), colors):
        edges = list(sol.used_edges)
        if edges:
            nx.draw_networkx_edges(net.G, pos, edgelist=edges, 
                                   edge_color=color, width=3, alpha=0.5, label=name)
            
    # 3. Machines & Sources avec numérotation
    # On parcourt les paires pour mettre les numéros
    for i, (m, s, e) in enumerate(net.demand_pairs):
        # Machine (Carré)
        plt.text(pos[m][0], pos[m][1]+0.5, str(i), fontsize=12, fontweight='bold', color='magenta')
        # Source (Étoile)
        plt.text(pos[s][0], pos[s][1]+0.5, str(i), fontsize=12, fontweight='bold', color='orange')

    machines = [n for n in net.G.nodes if net.G.nodes[n].get('type') == 'machine']
    sources = [n for n in net.G.nodes if net.G.nodes[n].get('type') == 'source']
    
    nx.draw_networkx_nodes(net.G, pos, nodelist=machines, node_size=150, node_color='magenta', node_shape='s')
    nx.draw_networkx_nodes(net.G, pos, nodelist=sources, node_size=250, node_color='orange', node_shape='*')

    plt.title("Comparaison : Dijkstra (Bleu) vs Density (Orange)", fontsize=15)
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Comparaison sauvegardée : {save_path}")
    plt.show()

def main():
    nom_du_graphe = "graphe_1"
    base_dir = "tests_2/graphs"
    
    cout_var = 1.0
    cout_fix = 10.0
    evaluator = Evaluator(weight_variable=cout_var, weight_fixed=cout_fix)

    # --- 1. GESTION DU GRAPHE ---
    path_pkl = Path(base_dir) / nom_du_graphe / f"{nom_du_graphe}.pkl"

    if path_pkl.exists():
        print(f"📂 Chargement : {nom_du_graphe}")
        net = RobustNetwork.load(nom_du_graphe, base_dir=base_dir)
    else:
        print("🛠️ Création d'un nouveau graphe...")
        net = RobustNetwork(size_x=20, size_y=20, n_nodes=20, n_segments=40, n_pairs=5)
        net.generate()
        net.resolve_colinear_overlaps()
        net.simplify_collinear_nodes()
        net.subdivide_long_edges()
        net.generate_demand_pairs()
        net.save(nom_du_graphe, base_dir=base_dir) 

    # --- 2. RÉSOLUTION : DIJKSTRA ---
    print("\n🚀 Lancement Naive Solver...")
    naive_solver = NaiveSolver(net)
    naive_solution = naive_solver.solve()
    metrics_n = print_compact_score("Dijkstra", naive_solution, evaluator)
    
    p_naive = Path(base_dir) / nom_du_graphe / "solution_naive.png"
    naive_solution.plot(save_path=p_naive, show=False, info_text=get_info_text("Dijkstra", metrics_n))

    # --- 3. RÉSOLUTION : DENSITY ---
    print("\n🚀 Lancement Density Solver...")
    density_solver = DensitySolver(net, fixed_cost_per_m=cout_fix, variable_cost_factor=cout_var, lambda_factor=1.5)
    density_solution = density_solver.solve()
    metrics_d = print_compact_score("Density", density_solution, evaluator)
    
    p_density = Path(base_dir) / nom_du_graphe / "solution_density.png"
    density_solution.plot(save_path=p_density, show=False, info_text=get_info_text("Density", metrics_d))

    # --- 4. COMPARAISON FUSIONNÉE ---
    p_comp = Path(base_dir) / nom_du_graphe / "comparaison_globale.png"
    comparaison = {
        "Dijkstra": naive_solution,
        "Density": density_solution
    }
    plot_comparison(net, comparaison, save_path=p_comp)

if __name__ == "__main__":
    main()