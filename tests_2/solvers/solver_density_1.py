import networkx as nx
from objects.solution import RoutingSolution
from objects.graph import RobustNetwork

class DensitySolver:
    def __init__(self, network: RobustNetwork, fixed_cost_per_m=10.0, variable_cost_factor=1.0, lambda_factor=2.0):
        """
        Le solveur a besoin de connaître les pondérations de coûts pour prendre 
        ses décisions de routage (même si c'est l'Evaluator qui fera la facture finale).
        """
        self.network = network
        self.fixed_cost = fixed_cost_per_m
        self.var_cost = variable_cost_factor
        self.lambda_factor = lambda_factor

    def solve(self):
        solution = RoutingSolution(self.network)
        
        if not hasattr(self.network, 'demand_pairs') or not self.network.demand_pairs:
            print("Aucune paire Machine/Source trouvée.")
            return solution

        # ---------------------------------------------------------
        # 1. INITIALISATION DU CHAMP DE GRAVITÉ
        # ---------------------------------------------------------
        self.network.compute_density_field(self.network.demand_pairs, lambda_factor=self.lambda_factor)

        # ---------------------------------------------------------
        # 2. TRI DES CÂBLES (Heuristique : Moment Énergétique E * L)
        # ---------------------------------------------------------
        def calculate_moment(pair):
            m_node, s_node, energy = pair
            pos_m = self.network.G.nodes[m_node]['pos']
            pos_s = self.network.G.nodes[s_node]['pos']
            # Distance euclidienne (à vol d'oiseau)
            dist = ((pos_m[0]-pos_s[0])**2 + (pos_m[1]-pos_s[1])**2)**0.5
            return energy * dist

        # On trie par le produit Energie * Distance estimée
        sorted_demands = sorted(self.network.demand_pairs, key=calculate_moment, reverse=True)

        # ---------------------------------------------------------
        # 3. ROUTAGE ITÉRATIF
        # ---------------------------------------------------------
        for machine, source, energy in sorted_demands:
            
            # A. Mise à jour dynamique des poids du graphe pour CETTE itération
            for u, v in self.network.G.edges():
                length = self.network.G[u][v]['weight']
                density = self.network.G[u][v].get('density', 0)
                
                # Le coût variable s'applique toujours (on paie le cuivre)
                cost_var = length * energy * self.var_cost
                
                # Vérification : la tranchée est-elle déjà ouverte ?
                edge_tuple = tuple(sorted((u, v)))
                
                if edge_tuple in solution.used_edges:
                    # BINGO ! La tranchée est déjà là, le coût fixe est nul.
                    cost_fixe = 0.0 
                else:
                    # La tranchée est fermée. On applique la magie de la densité :
                    # Plus la densité est forte, plus ce coût perçu s'effondre.
                    cost_fixe_brut = length * self.fixed_cost
                    cost_fixe = cost_fixe_brut / (1.0 + density)
                    
                # On crée un attribut temporaire spécialement pour le Dijkstra actuel
                self.network.G[u][v]['current_routing_cost'] = cost_var + cost_fixe

            # B. Lancement de Dijkstra avec ces poids distordus
            try:
                path = nx.shortest_path(
                    self.network.G, 
                    source=machine, 
                    target=source, 
                    weight='current_routing_cost'
                )
                
                # C. Ajout à la solution (ce qui ajoute automatiquement les arêtes à used_edges)
                solution.add_route(machine, source, energy, path)
                
            except nx.NetworkXNoPath:
                print(f"❌ Impossible de relier {machine} à {source}.")

        return solution