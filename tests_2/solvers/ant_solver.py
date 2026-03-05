import networkx as nx
import random

from objects.graph import RobustNetwork
from objects.solution import RoutingSolution

class AntSolver:
    def __init__(self, network: RobustNetwork, fixed_cost_per_m=10.0, variable_cost_factor=1.0, 
                 n_ants=20, n_iterations=30):
        self.network = network
        self.fixed_cost = fixed_cost_per_m
        self.var_cost = variable_cost_factor
        
        # Paramètres de la colonie
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = 0.1  # Évaporation à chaque tour
        self.alpha = 1.0  # Importance des phéromones
        
        # Initialisation des phéromones de base (1.0 partout)
        self.pheromones = {(u, v): 1.0 for u, v in self.network.G.edges()}

    def solve(self):
        best_overall_solution = None
        best_overall_score = float('inf')

        print(f"AntSolver (Pure ACO): Running {self.n_iterations} iterations...")

        for iteration in range(self.n_iterations):
            solutions_iteration = []
            
            for _ in range(self.n_ants):
                ant_solution = self._build_ant_solution()
                score = self._quick_evaluate(ant_solution)
                solutions_iteration.append((ant_solution, score))
                
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_solution = ant_solution

            self._update_pheromones(solutions_iteration)
            
            if iteration % 5 == 0:
                print(f"  Iteration {iteration}: Best Score = {best_overall_score:.2f}")

        return best_overall_solution

    def _build_ant_solution(self):
        """Une fourmi trace un chemin pour chaque paire demandée."""
        solution = RoutingSolution(self.network)
        
        # Ordre de routage aléatoire pour explorer différentes combinaisons
        pairs = list(self.network.demand_pairs)
        random.shuffle(pairs)

        for machine, source, energy in pairs:
            # Recherche de chemin influencée uniquement par les phéromones
            path = self._find_path_for_ant(machine, source, energy, solution.used_edges)
            solution.add_route(machine, source, energy, path)
            
        return solution

    def _find_path_for_ant(self, start, end, energy, global_used_edges):
        """Dijkstra probabiliste basé UNIQUEMENT sur Phéromones + Coût."""
        temp_graph = self.network.G.copy()
        
        for u, v in temp_graph.edges():
            length = temp_graph[u][v]['weight']
            # On récupère le niveau de phéromone de l'arête (indépendant du sens)
            pheromone = self.pheromones.get(tuple(sorted((u, v))), 1.0)
            
            # Coût variable (le cuivre, proportionnel à l'énergie)
            c_var = length * energy * self.var_cost
            
            # Coût fixe (la tranchée)
            if tuple(sorted((u, v))) in global_used_edges:
                c_fixe = 0 # Tranchée déjà ouverte par cette fourmi
            else:
                # La force d'attraction ne dépend QUE de la trace laissée par les fourmis précédentes
                bias = (pheromone ** self.alpha)
                c_fixe = (length * self.fixed_cost) / bias
            
            temp_graph[u][v]['ant_cost'] = c_var + c_fixe

        # Les fourmis essaient de minimiser ce coût déformé
        return nx.shortest_path(temp_graph, start, end, weight='ant_cost')

    def _update_pheromones(self, solutions_iteration):
        """Évaporation et dépôt de nouvelles phéromones."""
        # 1. Évaporation (oublier les mauvaises pistes)
        for edge in self.pheromones:
            self.pheromones[edge] *= (1.0 - self.evaporation_rate)
        
        # 2. Dépôt (récompenser les meilleures fourmis de l'itération)
        solutions_iteration.sort(key=lambda x: x[1])
        top_ants = max(1, len(solutions_iteration) // 5) # Top 20%
        
        for i in range(top_ants):
            sol, score = solutions_iteration[i]
            # Formule de récompense : on donne beaucoup si le score est bas
            reward = 1000.0 / (score + 1) 
            
            for edge in sol.used_edges:
                # L'arête doit être stockée de façon unique (tuple trié)
                sorted_edge = tuple(sorted(edge))
                if sorted_edge in self.pheromones:
                    self.pheromones[sorted_edge] += reward

    def _quick_evaluate(self, solution):
        """Évaluation rapide du score pour le classement interne des fourmis."""
        total_fixed = sum(self.network.G[u][v]['weight'] for u, v in solution.used_edges) * self.fixed_cost
        total_var = sum(length * e * self.var_cost for m, s, e, length, path in solution.routes)
        return total_fixed + total_var