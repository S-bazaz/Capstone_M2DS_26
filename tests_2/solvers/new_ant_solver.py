import numpy as np
import networkx as nx
import random


import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.image as mpimg


from objects.graph import RobustNetwork
from objects.solution import RoutingSolution

from pathlib import Path

graph_name = "graph_test_2"

# 1. Génération
# net = RobustNetwork(n_nodes = 15, n_segments=40, n_pairs=5)
# net.generate()
# net.resolve_colinear_overlaps()
# net.simplify_collinear_nodes()
# net.generate_demand_pairs()
# net.save(graph_name)

graph: RobustNetwork = RobustNetwork.load(graph_name)
graph.dist_dict = dict(nx.all_pairs_dijkstra_path_length(graph.G, weight='weight'))


n_edge = graph.G.number_of_edges()



####### Begin Param ######
n_ants = 300
half_life = 100
starting_pheromone = 1
era_length = 5000
n_era = 101
######## END Param ########

####### Begin constants ######
min_phero = starting_pheromone/10
threshold_suppresion = starting_pheromone/11
threshold_suppresion_target = starting_pheromone/9
decay_factor = np.exp(-np.log(2)/half_life)
pheromone_limit = n_edge * starting_pheromone
pheromone_gain = pheromone_limit*(1-decay_factor)/n_ants
######## END constants ########

l = 2

print(graph.G.number_of_edges())
print(f"decay_factor value is {decay_factor}")
print(f"pheromone_gain value is {pheromone_gain}")
print(f"pheromone_limit value is {pheromone_limit}")








def add_pheromones(graph: RobustNetwork):
    for u, v, edge in list(graph.G.edges(data=True)):
        edge["pheromone"] = starting_pheromone

def cost(graph: RobustNetwork, idx_1, idx_2):
    # Return the cost of linking idx_1 and idx_2
    c = graph.dist_dict[idx_1][idx_2]
    return c

class Ant_Worker:
    def __init__(self, graph: RobustNetwork, idx_start, idx_goal):
        self.graph = graph
        self.source_idx = idx_start
        self.machine_idx= idx_goal
        self.idx = idx_start
        self.past_idx = None
        
        self.start_idx = idx_start
        self.objective_idx = idx_goal
        self.objective = self.graph.G.nodes[idx_goal]

        #remove_pheromone_mode = True if we had no choice but to turn back.
        self.past = [self.idx]
        self.seen_nodes = [self.idx]
        self.remove_pheromone_mode = False
    
    def ponderation(self, neighbor_idx, pr):
        edge = self.graph.G[self.idx][neighbor_idx]

        neighbor_d_objective = self.graph.dist_dict[neighbor_idx][self.objective_idx]
        d_obj = neighbor_d_objective
        pheromones = edge["pheromone"]
        edge_cost = cost(self.graph, neighbor_idx, self.idx)

        return 1.0 / (d_obj + edge_cost + 1e-6) * pheromones**(pr)

    def check_obj_reached(self):
        if self.idx == self.objective_idx:
            self.objective_idx = self.start_idx
            self.start_idx = self.idx
            self.past_idx = None
            self.seen_nodes = [self.idx]
            self.past = [self.idx]

    def chose_movement(self, pr):
        neightbors_idxs = list(i for i in self.graph.G.neighbors(self.idx) if i not in self.seen_nodes)
        if len(neightbors_idxs) == 0:
            self.remove_pheromone_mode = True
            if len(self.past) > 1:
                self.past.pop()
                return self.past[-1]
            return self.idx
        else:
            self.remove_pheromone_mode = False
            l_choice = [self.ponderation(neightbor_idx, pr) for neightbor_idx in neightbors_idxs]
            choice_idx = random.choices(neightbors_idxs, weights=l_choice, k=1)[0]
            self.past.append(choice_idx)
            self.seen_nodes.append(choice_idx)
            return choice_idx
    
    def add_pheromones(self, idx_1, idx_2):
        if idx_1 != idx_2:
            self.graph.G[idx_1][idx_2]["pheromone"] += pheromone_gain

    def remove_pheromones(self, idx_1, idx_2):
        l_seen_nodes = len(self.seen_nodes)
        turn_passed = 0
        for i in range(l_seen_nodes):
            if self.seen_nodes[l_seen_nodes- 1 - i] == idx_1:
                turn_passed = i
                break
        if idx_1 != idx_2:
            self.graph.G[idx_1][idx_2]["pheromone"] = max(min_phero, self.graph.G[idx_1][idx_2]["pheromone"] - pheromone_gain*decay_factor**turn_passed)

    def take_turn(self, pr):
        chosen_idx = self.chose_movement(pr)
        self.past_idx = self.idx
        self.idx = chosen_idx

        if self.remove_pheromone_mode:
            self.remove_pheromones(self.idx, self.past_idx)
        else:
            self.add_pheromones(self.idx, self.past_idx)
        self.check_obj_reached()




class World:
    def __init__(self, graph: RobustNetwork, n_ants = 1):
        self.graph = graph
        self.n_ants = n_ants
        self.ant_list : list[Ant_Worker] = []
        self.year = 0
        self.era = 0
        self.total_era = n_era
        self.l = 0
        self.total_phero = None
        self.threshold = threshold_suppresion
        self.threshold_step = (threshold_suppresion_target - threshold_suppresion)/self.total_era
        self.deleted_edges_pool = []
        self.n_restore_max = 3

    def create_world(self):
        add_pheromones(self.graph)
        s = sum([(pair[2]*self.graph.dist_dict[pair[0]][pair[1]])**1.5 for pair in self.graph.demand_pairs])
        l_ants = [int((self.n_ants/s*pair[2]*self.graph.dist_dict[pair[0]][pair[1]])**1.5) for pair in self.graph.demand_pairs]

        for i in range(len(l_ants)):
            for _ in range(l_ants[i]):
                self.ant_list.append(Ant_Worker(self.graph, self.graph.demand_pairs[i][0], self.graph.demand_pairs[i][1]))
        # print(f"Total of {len(self.ant_list)} ants created")
            

    def next_era(self):
        candidates = [
            (u, v) for u, v, data in self.graph.G.edges(data=True) 
            if data["pheromone"] < self.threshold
        ]
        
        candidates.sort(key=lambda edge: self.graph.G[edge[0]][edge[1]]["pheromone"])
        
        for u, v in candidates:
            edge_data = self.graph.G[u][v].copy()
            self.graph.G.remove_edge(u, v)
            
            is_safe = True
            for source, machine, _ in self.graph.demand_pairs:
                if not nx.has_path(self.graph.G, source, machine):
                    is_safe = False
                    break
            
            if not is_safe:
                self.graph.G.add_edge(u, v, **edge_data)
            else:
                self.deleted_edges_pool.append((u, v, edge_data))

        n_to_restore = int(self.n_restore_max*self.era/n_era)
        if self.deleted_edges_pool and n_to_restore > 0:
            to_restore = random.sample(
                self.deleted_edges_pool, 
                min(n_to_restore, len(self.deleted_edges_pool))
            )
            
            for u, v, data in to_restore:
                data["pheromone"] = starting_pheromone 
                self.graph.G.add_edge(u, v, **data)
                self.deleted_edges_pool = [
                    e for e in self.deleted_edges_pool 
                    if not (e[0] == u and e[1] == v)
                ]

        self.ant_list = []
        self.era += 1
        self.year = 0
        self.l = 0
        self.threshold += self.threshold_step
        
        self.graph.dist_dict = dict(nx.all_pairs_dijkstra_path_length(self.graph.G, weight='weight'))
        self.create_world()


    def decay(self):
        self.total_phero = 0
        for u, v, edge in list(self.graph.G.edges(data=True)):
            edge["pheromone"] *= decay_factor
            self.total_phero += edge["pheromone"]
 
    def take_turn(self):
        for ant in self.ant_list:
            ant.take_turn(self.l)
        self.decay()
        self.year += 1
        self.l = (self.year / era_length) * l
    
    def plot(self, threshold=pheromone_limit/2000):
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        pos = nx.get_node_attributes(self.graph.G, 'pos')
        
        filtered_edges = []
        filtered_pheromones = []
        
        for u, v, data in self.graph.G.edges(data=True):
            p = data.get("pheromone", starting_pheromone)
            if p >= threshold:
                filtered_edges.append((u, v))
                filtered_pheromones.append(p)
                
        if not filtered_edges:
            print(f"Aucune arête au-dessus du seuil de {threshold}.")
            return
            
        min_phero = min(filtered_pheromones)
        max_phero = max(filtered_pheromones)
        
        if max_phero == min_phero:
            max_phero += 1e-6

        all_vals = np.array(filtered_pheromones)
        bounds = [
                0,
                self.threshold,        # Zone de nettoyage (juste au dessus du minimum)
                np.percentile(all_vals, 40),   # Le point d'équilibre initial (Zone neutre)
                np.percentile(all_vals, 75), # Top 25% des routes
                np.percentile(all_vals, 90), # Top 10% (Les autoroutes)
                max(all_vals.max(), 5) if len(all_vals)>0 else 10      # Le maximum actuel
            ]
        bounds = np.unique(np.sort(bounds))

        colors = ['#87cefa', '#ffd700', '#ff8c00', '#d73027', '#8b0000', "#070101"]
        custom_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
        
        mapped_colors = [custom_cmap(norm(p)) for p in filtered_pheromones]
        
        nx.draw_networkx_edges(
            self.graph.G, pos, 
            edgelist=filtered_edges,
            edge_color=mapped_colors,
            width=3,
            alpha=0.9,
            ax=ax
        )   
        
        nx.draw_networkx_nodes(self.graph.G, pos, node_size=30, node_color='gray', ax=ax)
        
        machines = [n for n in self.graph.G.nodes if self.graph.G.nodes[n].get('type') == 'machine']
        sources = [n for n in self.graph.G.nodes if self.graph.G.nodes[n].get('type') == 'source']
        
        nx.draw_networkx_nodes(self.graph.G, pos, nodelist=machines, node_size=150, node_color='magenta', node_shape='s', label='Machines', ax=ax)
        nx.draw_networkx_nodes(self.graph.G, pos, nodelist=sources, node_size=200, node_color='orange', node_shape='*', label='Sources', ax=ax)
        
        sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Concentration de Phéromones', ticks=bounds, format="%.1f")
        
        plt.title(f"Simulation Multi-Agents : Autoroutes (Seuil >= {round(threshold,2)})")
        plt.axis('off') 
        plt.legend(loc="upper right")
        plt.savefig(f"graphs/{graph_name}/{self.era}.png", dpi=300, bbox_inches='tight')
        plt.close()


    def get_subgraph(self):
        ...


world = World(graph, n_ants= n_ants)
world.create_world()
for i in range(n_era):
    print(f"Starting era {i}")
    for j in range(era_length):
        world.take_turn()
    world.plot()
    world.next_era()





def plot_progression(folder="results", pattern="graph_era_*.png"):
    # 1. Récupérer et trier les fichiers (important pour l'ordre chronologique)
    path_list = sorted(Path(folder).glob(pattern), 
                       key=lambda x: int(''.join(filter(str.isdigit, x.name)) or 0))
    
    if len(path_list) == 0:
        print("Aucune image trouvée.")
        return

    # 2. Créer la grille (2 lignes, 5 colonnes pour 10 images)
    n_imgs = len(path_list)
    cols = 5
    rows = (n_imgs + cols - 1) // cols  # Calcule le nombre de lignes nécessaires
    
    fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
    axes = axes.flatten() # On aplatit la grille pour boucler facilement

    for i, img_path in enumerate(path_list):
        img = mpimg.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Étape {i+1}", fontsize=14)
        axes[i].axis('off') # Cache les axes (numéros de pixels)

    # 3. Cacher les sous-plots vides si on a moins de 10 images
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    # Tu peux aussi faire plt.savefig("progression_totale.png")

# Appel de la fonction
plot_progression()


# python -m solvers.new_ant_solver