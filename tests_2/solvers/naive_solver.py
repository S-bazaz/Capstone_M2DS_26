import networkx as nx
from objects.graph import RobustNetwork
from objects.solution import RoutingSolution

class NaiveSolver:
    def __init__(self, network: RobustNetwork):
        """
        Prend en entrée l'infrastructure réseau (RobustNetwork) qui contient 
        le graphe G et la liste des paires (demand_pairs).
        """
        self.network = network

    def solve(self):
        """
        Exécute l'algorithme de routage et renvoie un objet RoutingSolution.
        Ici : le plus court chemin classique (Dijkstra), câble par câble.
        """
        # On initialise une solution vide
        solution = RoutingSolution(self.network)
        
        # S'il n'y a pas de paires à relier, on renvoie la solution vide
        if not hasattr(self.network, 'demand_pairs') or not self.network.demand_pairs:
            print("⚠️ Aucune paire Machine/Source trouvée dans le réseau.")
            return solution

        # On boucle sur chaque besoin en énergie
        for machine, source, energy in self.network.demand_pairs:
            try:
                # NetworkX calcule le plus court chemin basé sur l'attribut 'weight' (la distance)
                path = nx.shortest_path(
                    self.network.G, 
                    source=machine, 
                    target=source, 
                    weight='weight'
                )
                
                # On ajoute ce chemin à notre solution
                solution.add_route(machine, source, energy, path)
                
            except nx.NetworkXNoPath:
                print(f"❌ Impossible de relier la Machine {machine} à la Source {source}.")

        return solution