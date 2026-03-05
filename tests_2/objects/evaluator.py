class Evaluator:
    def __init__(self, weight_variable=1.0, weight_fixed=10.0):
        self.weight_variable = weight_variable
        self.weight_fixed = weight_fixed

    def evaluate(self, solution):
        """
        Calcule les scores. Compatible avec solution.routes (liste de tuples).
        """
        network = solution.network
        total_variable_cost = 0.0
        
        # On parcourt la liste des routes
        for r in solution.routes:
            # Structure du tuple : (machine, source, energy, length, path)
            # Donc energy = index 2, path = index 4
            energy = r[2]
            path = r[4]
            
            path_length = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                path_length += network.G[u][v]['weight']
                
            total_variable_cost += path_length * energy
            
        total_fixed_cost = len(solution.used_edges)
        
        weighted_variable = self.weight_variable * total_variable_cost
        weighted_fixed = self.weight_fixed * total_fixed_cost
        
        return {
            'variable_raw': total_variable_cost,
            'fixed_raw': total_fixed_cost,
            'weighted_variable': weighted_variable,
            'weighted_fixed': weighted_fixed,
            'total_score': weighted_variable + weighted_fixed
        }