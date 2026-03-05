import numpy as np
import networkx as nx
import random

from objects.graph import RobustNetwork
from objects.solution import RoutingSolution

class AntDensitySolver:
    def __init__(self, network: RobustNetwork, fixed_cost_per_m=10.0, variable_cost_factor=1.0, 
                 lambda_factor=1.5, n_ants=20, n_iterations=30):
        self.network = network
        self.fixed_cost = fixed_cost_per_m
        self.var_cost = variable_cost_factor
        self.lambda_factor = lambda_factor
        
        # Paramètres de la colonie
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = 0.1  # Pourcentage de phéromone qui s'évapore
        self.alpha = 1.0  # Importance des phéromones (mémoire)
        self.gamma = 2.0  # Importance de la densité (biais géographique)
        
        # Initialisation des phéromones
        self.pheromones = {(u, v): 1.0 for u, v in self.network.G.edges()}

    def solve(self):
        # 1. On calcule la densité a priori (ton algorithme initial)
        self.network.compute_density_field(self.network.demand_pairs, lambda_factor=self.lambda_factor)
        
        best_overall_solution = None
        best_overall_score = float('inf')

        print(f"AntDensitySolver: Running {self.n_iterations} iterations...")

        for iteration in range(self.n_iterations):
            solutions_iteration = []
            
            # 2. Chaque fourmi construit UNE solution complète pour TOUTES les paires
            for _ in range(self.n_ants):
                ant_solution = self._build_ant_solution()
                # On évalue la solution de la fourmi (on simule l'Evaluator)
                score = self._quick_evaluate(ant_solution)
                solutions_iteration.append((ant_solution, score))
                
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_solution = ant_solution

            # 3. Mise à jour des phéromones (Évaporation + Dépôt)
            self._update_pheromones(solutions_iteration)
            
            if iteration % 5 == 0:
                print(f"  Iteration {iteration}: Best Score = {best_overall_score:.2f}")

        return best_overall_solution

    def _build_ant_solution(self):
        """Une fourmi trace un chemin pour chaque paire demandée."""
        solution = RoutingSolution(self.network)
        
        # On mélange l'ordre des paires pour favoriser l'exploration
        pairs = list(self.network.demand_pairs)
        random.shuffle(pairs)

        for machine, source, energy in pairs:
            # Calcul des poids "perçus" par la fourmi
            path = self._find_path_for_ant(machine, source, energy, solution.used_edges)
            solution.add_route(machine, source, energy, path)
            
        return solution

    def _find_path_for_ant(self, start, end, energy, global_used_edges):
        """Construction du chemin nœud par nœud avec probabilités proportionnelles."""
        path = [start]
        current = start
        visited = set([start])
        
        # Position de la destination pour l'heuristique directionnelle
        pos_end = np.array(self.network.G.nodes[end]['pos'])

        while current != end:
            # 1. Identifier les voisins valides (pas encore visités)
            neighbors = [n for n in self.network.G.neighbors(current) if n not in visited]
            
            if not neighbors:
                # ÉCHEC : La fourmi est dans un cul-de-sac.
                # Solution de secours : on la remet sur le droit chemin avec Dijkstra
                # (Dans un ACO pur, la fourmi "meurt", mais ici on veut une solution valide)
                rescue_path = nx.shortest_path(self.network.G, current, end, weight='weight')
                return path[:-1] + rescue_path

            # 2. Calculer l'attractivité de chaque voisin
            attractiveness = []
            for neighbor in neighbors:
                edge = tuple(sorted((current, neighbor)))
                
                length = self.network.G[current][neighbor]['weight']
                density = self.network.G[current][neighbor].get('density', 0)
                pheromone = self.pheromones.get(edge, 1.0)
                
                # Calcul du coût local
                c_var = length * energy * self.var_cost
                c_fixe = 0 if edge in global_used_edges else (length * self.fixed_cost)
                cost = c_var + c_fixe
                
                # Heuristique : On ajoute la distance restante pour guider la fourmi
                pos_n = np.array(self.network.G.nodes[neighbor]['pos'])
                dist_to_end = np.linalg.norm(pos_n - pos_end)
                
                # L'heuristique globale (eta) : inversement proportionnelle au coût + distance
                eta = 1.0 / (cost + dist_to_end + 1e-6)
                
                # Formule de transition de l'ACO
                score = (pheromone ** self.alpha) * (eta ** 2.0) * ((1.0 + density) ** self.gamma)
                attractiveness.append(score)
            
            # 3. Choix probabiliste (Roulette Wheel Selection)
            total_attr = sum(attractiveness)
            if total_attr == 0: # Sécurité mathématique
                probabilities = [1.0 / len(neighbors)] * len(neighbors)
            else:
                probabilities = [a / total_attr for a in attractiveness]
            
            # La fourmi choisit son prochain pas !
            next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
            
            # 4. On avance
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
        return path
    def _update_pheromones(self, solutions_iteration):
        # Évaporation
        for edge in self.pheromones:
            self.pheromones[edge] *= (1.0 - self.evaporation_rate)
        
        # Dépôt (Seules les bonnes fourmis renforcent les pistes)
        # On prend le top 20% des solutions de l'itération
        solutions_iteration.sort(key=lambda x: x[1])
        for i in range(len(solutions_iteration) // 5):
            sol, score = solutions_iteration[i]
            reward = 1000.0 / (score + 1) # Plus le score est bas, plus la récompense est forte
            for edge in sol.used_edges:
                if edge in self.pheromones:
                    self.pheromones[edge] += reward

    def _quick_evaluate(self, solution):
        """Version simplifiée de l'Evaluator pour un calcul rapide en interne."""
        # Coût fixe
        total_fixed = sum(self.network.G[u][v]['weight'] for u, v in solution.used_edges) * self.fixed_cost
        # Coût variable
        total_var = 0
        for m, s, e, length, path in solution.routes:
            total_var += length * e * self.var_cost
        return total_fixed + total_var