import numpy as np
import pandas as pd
from objects.graph import RobustNetwork
from solvers.naive_solver import NaiveSolver
from solvers.solver_density_1 import DensitySolver
from solvers.ant_solver import AntSolver
from solvers.ant_density_solver import AntDensitySolver
from objects.evaluator import Evaluator

# python -m tests_2.stats

def run_benchmark():
    n_graphs = 10
    fixed_costs = [1, 5, 10, 20, 50] 
    variable_cost = 1.0
    
    results = []

    print(f"BENCHMARK SYSTEM - Running {n_graphs} iterations per scenario")
    print("-" * 80)

    for c_fixe in fixed_costs:
        print(f"Scenario: Fixed Cost = {c_fixe} | Variable Cost = {variable_cost}")
        
        scores_naive = []
        scores_density = []
        scores_ant = []
        scores_ant_density = []
        
        for i in range(n_graphs):
            net = RobustNetwork(size_x=25, size_y=25, n_nodes=30, n_segments=80, n_pairs=10)
            net.generate()
            net.subdivide_long_edges()
            net.generate_demand_pairs()
            
            evaluator = Evaluator(weight_variable=variable_cost, weight_fixed=c_fixe)
            
            sol_naive = NaiveSolver(net).solve()
            scores_naive.append(evaluator.evaluate(sol_naive)['total_score'])
            
            sol_density = DensitySolver(
                net, fixed_cost_per_m=c_fixe, variable_cost_factor=variable_cost, lambda_factor=1.5
            ).solve()
            scores_density.append(evaluator.evaluate(sol_density)['total_score'])
            
            # sol_ant = AntSolver(
            #     net, fixed_cost_per_m=c_fixe, variable_cost_factor=variable_cost, n_ants=20, n_iterations=30
            # ).solve()
            # scores_ant.append(evaluator.evaluate(sol_ant)['total_score'])
            
            sol_ant_density = AntDensitySolver(
                net, fixed_cost_per_m=c_fixe, variable_cost_factor=variable_cost, lambda_factor=1.5, n_ants=20, n_iterations=30
            ).solve()
            scores_ant_density.append(evaluator.evaluate(sol_ant_density)['total_score'])
            
        mean_naive = np.mean(scores_naive)
        mean_density = np.mean(scores_density)
        mean_ant = np.mean(scores_ant)
        mean_ant_density = np.mean(scores_ant_density)

        results.append({
            "Ratio_F/V": c_fixe / variable_cost,
            "Avg_Dijkstra": mean_naive,
            "Avg_Density": mean_density,
            # "Avg_Ant": mean_ant,
            "Avg_AntDens": mean_ant_density,
            "Gain_Density": (mean_naive - mean_density) / mean_naive,
            "Gain_Ant": (mean_naive - mean_ant) / mean_naive,
            "Gain_AntDens": (mean_naive - mean_ant_density) / mean_naive
        })

    df = pd.DataFrame(results)
    
    print("\nSUMMARY OF PERFORMANCE METRICS")
    print("=" * 105)
    
    formatters = {
        "Avg_Dijkstra": "{:,.2f}".format,
        "Avg_Density": "{:,.2f}".format,
        # "Avg_Ant": "{:,.2f}".format,
        "Avg_AntDens": "{:,.2f}".format,
        "Gain_Density": "{:.2%}".format,
        "Gain_Ant": "{:.2%}".format,
        "Gain_AntDens": "{:.2%}".format
    }
    
    print(df.to_string(index=False, formatters=formatters))
    print("=" * 105)

if __name__ == "__main__":
    run_benchmark()