from .geometry import Point, Points, Segment, Segments
import matplotlib.pyplot as plt
import random
import pickle
from pathlib import Path
import os 

class Coupling:
    def __init__(self, p1: Point, p2: Point, w: float):
        if w < 0:
            raise ValueError
        self.energy = p1
        self.machine = p2
        self.w = w
        

def default_f(_):
    return 1

class DynamicGraph():
    def __init__(self, x_size=10, y_size=10,
                n_node=10, n_segments=20, n_machine = 3,
                f = default_f,
                auto_gen = True,
                name = "test"):
        self.x_size = x_size
        self.y_size = y_size
        self.n_node = n_node
        self.n_segments = n_segments
        self.n_machine = n_machine
        self.point_list = Points()
        self.segment_list = Segments()
        self.coupling: list[Coupling] = []
        self.dist_dict = {}
        self.f = f
        self.name = name
        self.path = Path("graphs") / self.name
        if auto_gen:
            self.generate_pipeline()

    def generate_pipeline(self):
        for _ in range(self.n_node):
            self.point_list.add_point(round(random.uniform(0, self.x_size), 2), round(random.uniform(0, self.y_size), 2))

        for _ in range(self.n_segments):
            p1 = self.point_list.pick_unconnected_point()
            if not p1:
                p1 = self.point_list.pick_point()
            p2 = self.point_list.pick_point(exception_points=[p1])
            x, y = random.choice([(p1.x, p2.y), (p2.x, p1.y)])
            p3 = self.point_list.add_point(x,y)
            self.segment_list.add_segment(p1, p3, self.point_list)
            self.segment_list.add_segment(p3, p2, self.point_list)

        coupling_points = self.point_list.pick_points(2*self.n_machine)
        for i in range(self.n_machine):
            p1 = coupling_points[i]
            p2 = coupling_points[-(i+1)]
            w = random.randint(1, 8)
            #TODO @BaptisteMERESSE ameliorer la selection des w
            self.coupling.append(Coupling(p1, p2, w))

    def plot(self, show = True, save = False):
        plt.figure(figsize=(10, 10))
        
        for s in self.segment_list.l_segments:
            plt.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y], color='gray', alpha=0.3, zorder=1)
        
        xs = [p.x for p in self.point_list.l_points]
        ys = [p.y for p in self.point_list.l_points]
        plt.scatter(xs, ys, c='black', s=10, alpha=0.3, zorder=2)
        
        colors = plt.cm.tab10.colors 
        
        for i, c in enumerate(self.coupling):
            p1, p2 = c.energy, c.machine
            color = colors[i % len(colors)]
            
            plt.scatter([p1.x, p2.x], [p1.y, p2.y], 
                        color=color, s=80, edgecolors='black', 
                        linewidth=1.5, zorder=3, label=f'Pair {i}')
            
            plt.text(p1.x, p1.y + 0.2, f"{i}", fontsize=9, fontweight='bold', color=color)
            plt.text(p2.x, p2.y + 0.2, f"{i}", fontsize=9, fontweight='bold', color=color)

        plt.title(f"Dynamic Graph: {self.n_machine} Couplings")
        plt.xlim(-1, self.x_size + 1)
        plt.ylim(-1, self.y_size + 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.grid(visible=True, alpha = 0.3)
        if show:
            plt.show()
        if save:
            plt.savefig(self.path / "plot.png")

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.plot(show=False, save=True)
        data_path = self.path / "graph_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(name: str):
        path = Path("graphs") / name / "graph_data.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Le fichier {path} n'existe pas.")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def generate_live(self, delay=0.3):
        """Génère le réseau arête par arête avec affichage en temps réel."""
        plt.ion() # Active le mode interactif
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Génération et affichage des points initiaux
        for _ in range(self.n_node):
            self.point_list.add_point(
                round(random.uniform(0, self.x_size), 2), 
                round(random.uniform(0, self.y_size), 2)
            )
        
        # Affichage initial des points
        xs = [p.x for p in self.point_list.l_points]
        ys = [p.y for p in self.point_list.l_points]
        ax.scatter(xs, ys, c='red', s=20, zorder=3)
        ax.set_xlim(-1, self.x_size + 1)
        ax.set_ylim(-1, self.y_size + 1)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.title("Génération en direct...")
        # 2. Génération des segments un par un
        for _ in range(self.n_segments):
            p1 = self.point_list.pick_unconnected_point()
            if not p1:
                p1 = self.point_list.pick_point()
            
            p2 = self.point_list.pick_point_dist(p1, self.f)
            
            # Création du coude
            x, y = random.choice([(p1.x, p2.y), (p2.x, p1.y)])
            p3 = self.point_list.add_point(x, y)
            
            # Ajout des segments (la subdivision se fait en interne)
            self.segment_list.add_segment(p1, p3, self.point_list)
            self.segment_list.add_segment(p3, p2, self.point_list)

            # --- Mise à jour du graphique ---
            ax.clear() # On efface pour redessiner proprement
            
            # Redessiner la grille et les points
            ax.set_xlim(-1, self.x_size + 1)
            ax.set_ylim(-1, self.y_size + 1)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Tracer tous les segments actuels
            for s in self.segment_list.l_segments:
                ax.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y], 'b-', alpha=0.6)
            
            # Tracer tous les points (incluant les nouveaux points d'intersection)
            xs_live = [p.x for p in self.point_list.l_points]
            ys_live = [p.y for p in self.point_list.l_points]
            ax.scatter(xs_live, ys_live, c='red', s=20, zorder=3)
            
            plt.draw()
            plt.pause(delay) # Petite pause pour l'effet visuel

        plt.ioff()
        plt.show(block=True) 
        plt.close(fig)