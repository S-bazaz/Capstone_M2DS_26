import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

class RoutingSolution:
    def __init__(self, network):
        self.network = network
        self.routes = []      # Doit être une liste
        self.used_edges = set() # Doit être un set

    def add_route(self, machine, source, energy, path):
        # ... (ton code add_route précédent) ...
        path_length = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            path_length += self.network.G[u][v]['weight']
            edge = tuple(sorted((u, v)))
            self.used_edges.add(edge)
        
        self.routes.append((machine, source, energy, path_length, path))

    def plot(self, save_path=None, show=True, info_text=""):
        """Affiche le graphe avec les routes et les statistiques."""
        pos = nx.get_node_attributes(self.network.G, 'pos')
        plt.figure(figsize=(12, 10))
        
        # 1. Dessiner tout le réseau en fond (gris clair)
        nx.draw_networkx_edges(self.network.G, pos, edge_color='lightgray', alpha=0.3)
        nx.draw_networkx_nodes(self.network.G, pos, node_size=10, node_color='lightgray')
        
        # 2. Dessiner les arêtes utilisées (en noir épais)
        used_edges_list = list(self.used_edges)
        if used_edges_list:
            nx.draw_networkx_edges(self.network.G, pos, edgelist=used_edges_list, 
                                   edge_color='black', width=2.5, label="Supports")
            
        # 3. Dessiner les Machines (Carrés magenta) et Sources (Étoiles orange)
        machines = [n for n in self.network.G.nodes if self.network.G.nodes[n].get('type') == 'machine']
        sources = [n for n in self.network.G.nodes if self.network.G.nodes[n].get('type') == 'source']
        
        nx.draw_networkx_nodes(self.network.G, pos, nodelist=machines, node_size=150, 
                               node_color='magenta', node_shape='s', label="Machines")
        nx.draw_networkx_nodes(self.network.G, pos, nodelist=sources, node_size=200, 
                               node_color='orange', node_shape='*', label="Sources")

        # 4. Ajouter l'encadré de texte (Info Text)
        if info_text:
            plt.gca().text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                           fontsize=10, verticalalignment='top', family='monospace',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend(loc='lower right')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📸 Graphique sauvegardé : {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()