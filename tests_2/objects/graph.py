import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from pathlib import Path

class RobustNetwork:
    def __init__(self, size_x=10, size_y=10,
                n_nodes = 10, n_segments = 20,
                support_length = 1,
                n_pairs = 3):
        self.G = nx.Graph()
        self.size_x = size_x
        self.size_y = size_y
        self.n_nodes = n_nodes
        self.n_segments = n_segments
        self.n_pairs = n_pairs
        self.node_count = 0
        self.support_length = support_length

    def add_point(self, x, y, connected=False, is_initial=False):
        """
        Crée un nouveau point OU retourne l'ID d'un point existant 
        s'il y en a déjà un à ces coordonnées exactes.
        """
        # 1. On cherche si un point existe déjà à cette position
        for n in self.G.nodes:
            pos = self.G.nodes[n]['pos']
            # On utilise une petite tolérance (1e-5) pour éviter les bugs liés 
            # à l'arrondi des nombres à virgule (floats) en Python.
            if abs(pos[0] - x) < 1e-5 and abs(pos[1] - y) < 1e-5:
                # Le point existe ! On met à jour ses attributs si besoin
                if connected:
                    self.G.nodes[n]['connected'] = True
                if is_initial:
                    self.G.nodes[n]['is_initial'] = True

                # On retourne l'ID du noeud existant pour que les câbles s'y branchent
                return n

        # 2. Si aucun point ne correspond, on crée un nouveau noeud
        idx = self.node_count
        self.G.add_node(idx, pos=(x, y), connected=connected, is_initial=is_initial)
        self.node_count += 1
        return idx

    def get_pos(self, node_idx):
        return np.array(self.G.nodes[node_idx]['pos'])

    def calculate_dist(self, u, v):
        return np.linalg.norm(self.get_pos(u) - self.get_pos(v))

    def check_intersection(self, u1, v1, u2, v2):
        """Checks if segment (u1, v1) intersects (u2, v2)."""
        p1, p2 = self.get_pos(u1), self.get_pos(v1)
        p3, p4 = self.get_pos(u2), self.get_pos(v2)

        # 1. Identify which segment is Horizontal and which is Vertical
        # Compare IDs (integers) to avoid the NumPy ambiguity error
        h_nodes = None
        v_nodes = None

        if p1[1] == p2[1]: # Segment 1 is horizontal
            h_nodes, v_nodes = (u1, v1), (u2, v2)
        elif p3[1] == p4[1]: # Segment 2 is horizontal
            h_nodes, v_nodes = (u2, v2), (u1, v1)

        # 2. If we found one of each, check the bounds
        if h_nodes and v_nodes:
            # Get coordinates for the horizontal segment
            h_p1, h_p2 = self.get_pos(h_nodes[0]), self.get_pos(h_nodes[1])
            # Get coordinates for the vertical segment
            v_p1, v_p2 = self.get_pos(v_nodes[0]), self.get_pos(v_nodes[1])

            y_val = h_p1[1]
            x_val = v_p1[0]

            x_range = (min(h_p1[0], h_p2[0]), max(h_p1[0], h_p2[0]))
            y_range = (min(v_p1[1], v_p2[1]), max(v_p1[1], v_p2[1]))

            # Check if the vertical line's X is within the horizontal line's X-range
            # AND the horizontal line's Y is within the vertical line's Y-range
            if x_range[0] < x_val < x_range[1] and y_range[0] < y_val < y_range[1]:
                return (x_val, y_val)
     
        return None

    def add_robust_segment(self, u, v):
        """Equivalent to your Segments.add_segment with recursion."""
        if np.array_equal(self.get_pos(u), self.get_pos(v)):
            return

        intersection = None
        target_edge = None

        # NetworkX makes iterating through existing segments (edges) very easy
        for (edge_u, edge_v) in list(self.G.edges()):
            res = self.check_intersection(u, v, edge_u, edge_v)
            if res:
                intersection = res
                target_edge = (edge_u, edge_v)
                break

        if intersection:
            # 1. Create the new intersection node
            new_node = self.add_point(intersection[0], intersection[1], connected=True)

            # 2. Split the existing edge
            self.G.remove_edge(*target_edge)
            self.add_robust_segment(target_edge[0], new_node)
            self.add_robust_segment(target_edge[1], new_node)

            # 3. Add the two parts of the new segment
            self.add_robust_segment(u, new_node)
            self.add_robust_segment(v, new_node)
        else:
            # No intersection? Just add the edge with weight
            d = self.calculate_dist(u, v)
            self.G.add_edge(u, v, weight=d)
            self.G.nodes[u]['connected'] = True
            self.G.nodes[v]['connected'] = True

    def generate(self):
        # 1. Place initial machines
        n_nodes = self.n_nodes
        n_segments = self.n_segments 
        for _ in range(n_nodes+1):
            self.add_point(round(random.uniform(0, self.size_x), 2), 
                           round(random.uniform(0, self.size_y), 2),
                           is_initial=True)

        # 2. Generate connections
        for _ in range(n_segments+1):
            # Pick a starting point (unconnected if possible)
            unconnected = [n for n in self.G.nodes if not self.G.nodes[n]['connected']]
            p1 = random.choice(unconnected) if unconnected else random.choice(list(self.G.nodes))

            # Pick a destination
            others = [n for n in self.G.nodes if n != p1]
            p2 = random.choice(others)

            # Create the 'elbow' point (Manhattan routing)
            pos1, pos2 = self.get_pos(p1), self.get_pos(p2)
            corner_x, corner_y = random.choice([(pos1[0], pos2[1]), (pos2[0], pos1[1])])
            p3 = self.add_point(corner_x, corner_y)

            self.add_robust_segment(p1, p3)
            self.add_robust_segment(p3, p2)

    def subdivide_long_edges(self):
        max_length = self.support_length
        """
        Parcourt le graphe et découpe les arêtes dont la longueur dépasse max_length
        en n segments de taille égale (inférieure ou égale à max_length).
        """
        # ⚠️ CRUCIAL : On fige la liste des arêtes avec list() avant de boucler.
        # Sinon, modifier le graphe pendant qu'on le parcourt provoque une erreur.
        edges_to_check = list(self.G.edges())

        for u, v in edges_to_check:
            p1 = self.get_pos(u)
            p2 = self.get_pos(v)
            d = self.calculate_dist(u, v)

            if d > max_length:
                # 1. Calcul du nombre de sous-segments nécessaires
                n_segments = int(np.ceil(d / max_length))

                # 2. Suppression de l'arête d'origine (trop longue)
                self.G.remove_edge(u, v)

                # 3. Création des points intermédiaires et reconnexion
                previous_node = u
                for i in range(1, n_segments):
                    # Interpolation linéaire pour trouver les coordonnées X et Y
                    fraction = i / n_segments
                    new_x = p1[0] + fraction * (p2[0] - p1[0])
                    new_y = p1[1] + fraction * (p2[1] - p1[1])

                    # On ajoute le nouveau point de support au graphe
                    new_node = self.add_point(new_x, new_y, connected=True)

                    # On le connecte au point précédent avec le bon poids (distance)
                    sub_dist = self.calculate_dist(previous_node, new_node)
                    self.G.add_edge(previous_node, new_node, weight=sub_dist)

                    # Le nouveau point devient le "précédent" pour la prochaine itération
                    previous_node = new_node

                # 4. Ne pas oublier de connecter le tout dernier support au nœud d'arrivée (v)
                final_dist = self.calculate_dist(previous_node, v)
                self.G.add_edge(previous_node, v, weight=final_dist)

    def simplify_collinear_nodes(self, tolerance=1e-5):
        """
        Nettoie le graphe en supprimant les nœuds de degré 2 qui sont
        parfaitement alignés avec leurs voisins, fusionnant ainsi les arêtes.
        """
        nodes_removed = 0

        # On utilise une boucle while car la suppression d'un noeud modifie
        # potentiellement le degré de ses voisins (effet domino).
        while True:
            # On cherche les candidats : degré exactement égal à 2
            candidates = [n for n in self.G.nodes if self.G.degree(n) == 2]
            removed_in_this_pass = False

            for n in candidates:
                # Si le noeud a déjà été supprimé dans cette itération, on passe
                if n not in self.G:
                    continue

                # On récupère ses deux voisins
                neighbors = list(self.G.neighbors(n))
                if len(neighbors) != 2:
                    continue

                u, v = neighbors
                p_n = self.get_pos(n)
                p_u = self.get_pos(u)
                p_v = self.get_pos(v)

                # Vérification de l'alignement avec le produit vectoriel (cross product) 2D
                # Vecteur U->N et Vecteur N->V
                vec1 = p_n - p_u
                vec2 = p_v - p_n
                cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

                # Si le produit vectoriel est très proche de 0, les points sont alignés
                if abs(cross_product) < tolerance:
                    # On supprime le noeud intermédiaire
                    self.G.remove_node(n)

                    # On relie directement U et V avec le nouveau poids (distance totale)
                    new_dist = self.calculate_dist(u, v)
                    self.G.add_edge(u, v, weight=new_dist)

                    removed_in_this_pass = True
                    nodes_removed += 1

            # Si on n'a rien supprimé lors de ce passage complet, le graphe est propre !
            if not removed_in_this_pass:
                break

        print(f"Nettoyage terminé : {nodes_removed} nœuds superflus supprimés.")

    def point_to_segment_dist_vectorized(self, points, A, B):
        """
        Calcule la distance minimale entre un ensemble de points (N, 2)
        et un segment de droite défini par les points A et B.
        """
        AB = B - A
        AP = points - A

        # Produit scalaire pour trouver la projection orthogonale (vectorisé sur tous les points)
        dot_AP_AB = np.sum(AP * AB, axis=1)
        dot_AB_AB = np.sum(AB * AB)

        # Si A et B sont le même point, la distance est juste la norme jusqu'à A
        if dot_AB_AB == 0:
            return np.linalg.norm(points - A, axis=1)

        # Paramètre t de la projection (0 = point A, 1 = point B)
        t = dot_AP_AB / dot_AB_AB

        # On contraint t entre 0 et 1 pour ne pas déborder du segment (clip)
        t_clamped = np.clip(t, 0, 1)

        # Calcul des coordonnées du point le plus proche sur le segment
        # L'utilisation de [:, np.newaxis] permet de multiplier un vecteur (N,) avec (2,)
        closest_points = A + t_clamped[:, np.newaxis] * AB

        # On retourne la distance euclidienne entre les points et leurs projections
        return np.linalg.norm(points - closest_points, axis=1)

    def snap_and_merge(self, grid_size=0.5):
        """
        Aligne tous les nœuds sur une grille puis fusionne ceux qui tombent
        sur le même point. Garantit la conservation stricte des arêtes H/V.
        """
        # 1. Aimantation (Arrondi) des coordonnées sur la grille
        for n in self.G.nodes:
            pos = self.get_pos(n)
            # On arrondit au multiple de grid_size le plus proche
            snapped_x = round(pos[0] / grid_size) * grid_size
            snapped_y = round(pos[1] / grid_size) * grid_size
            self.G.nodes[n]['pos'] = (snapped_x, snapped_y)

        # 2. Regroupement des noeuds qui partagent la même nouvelle position
        pos_to_nodes = {}
        for n in list(self.G.nodes):
            pos = self.G.nodes[n]['pos']
            if pos not in pos_to_nodes:
                pos_to_nodes[pos] = []
            pos_to_nodes[pos].append(n)

        nodes_merged = 0

        # 3. Fusion des noeuds superposés
        for pos, nodes in pos_to_nodes.items():
            if len(nodes) > 1:
                # On garde le premier noeud comme "maître", on absorbe les autres
                primary_node = nodes[0]
                for other_node in nodes[1:]:
                    # Transférer les arêtes vers le noeud maître
                    for neighbor in list(self.G.neighbors(other_node)):
                        if neighbor != primary_node:
                            # La nouvelle distance sera parfaitement H ou V
                            dist = self.calculate_dist(primary_node, neighbor)
                            self.G.add_edge(primary_node, neighbor, weight=dist)

                    # Conserver le statut connecté si l'un d'eux l'était
                    if self.G.nodes[other_node].get('connected'):
                        self.G.nodes[primary_node]['connected'] = True

                    self.G.remove_node(other_node)
                    nodes_merged += 1

        # 4. Nettoyage final : on retire les boucles sur soi-même (self-loops)
        # qui apparaissent quand on fusionne les deux bouts d'un micro-segment
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        # 5. Mise à jour des poids (longueurs) de toutes les arêtes restantes
        for u, v in self.G.edges:
            self.G[u][v]['weight'] = self.calculate_dist(u, v)

        print(f"Aimantation sur grille de {grid_size} : {nodes_merged} micro-nœuds absorbés.")

    def resolve_colinear_overlaps(self, tolerance=1e-5):
        """
        Détecte si des nœuds sont posés directement sur des arêtes existantes
        (superposition colinéaire) et fractionne ces arêtes pour reconnecter
        la topologie proprement.
        """
        while True:
            split_happened = False
            edges = list(self.G.edges())
            nodes = list(self.G.nodes())

            for u, v in edges:
                if split_happened: 
                    break

                p_u = self.get_pos(u)
                p_v = self.get_pos(v)

                for n in nodes:
                    # On ignore les extrémités de l'arête
                    if n == u or n == v:
                        continue

                    p_n = self.get_pos(n)

                    # 1. Vérification de l'alignement (produit vectoriel)
                    vec1 = p_v - p_u
                    vec2 = p_n - p_u
                    cross_product = abs(vec1[0] * vec2[1] - vec1[1] * vec2[0])

                    if cross_product < tolerance:
                        # 2. Le point est aligné, mais est-il ENTRE u et v ?
                        # On utilise le produit scalaire (dot product). 
                        # S'il est positif, le point n est strictement entre u et v.
                        dot = np.dot(p_n - p_u, p_v - p_n)
                        if dot > 0:
                            # On a trouvé un noeud superposé sur l'arête !
                            # On casse la grande arête pour passer par ce noeud.
                            self.G.remove_edge(u, v)

                            dist1 = self.calculate_dist(u, n)
                            dist2 = self.calculate_dist(n, v)

                            self.G.add_edge(u, n, weight=dist1)
                            self.G.add_edge(n, v, weight=dist2)

                            split_happened = True
                            break  # On casse la boucle pour rafraîchir la liste des arêtes

            # Si on a scanné toutes les arêtes sans rien casser, c'est que le graphe est propre
            if not split_happened:
                break

    def generate_demand_pairs(self):
        """
        Sélectionne aléatoirement n_pairs parmi les points initiaux.
        Assigne un rôle (Machine ou Source) et une demande en énergie (1 à 10).
        """
        n_pairs = self.n_pairs
        # On récupère uniquement les noeuds qui ont été générés en premier
        initial_nodes = [n for n in self.G.nodes if self.G.nodes[n].get('is_initial')]

        # Sécurité : vérifier qu'on a assez de points initiaux
        max_pairs = len(initial_nodes) // 2
        if n_pairs > max_pairs:
            print(f"⚠️ Pas assez de noeuds initiaux pour {n_pairs} paires. Réduction à {max_pairs}.")
            n_pairs = max_pairs

        # On mélange pour tirer au sort
        random.shuffle(initial_nodes)

        self.demand_pairs = []

        for i in range(n_pairs):
            machine = initial_nodes[2*i]
            source = initial_nodes[2*i + 1]
            energy = random.randint(1, 10) # Quantité d'énergie (épaisseur du câble)

            self.demand_pairs.append((machine, source, energy))

            # On enregistre le rôle directement dans le graphe pour l'affichage
            self.G.nodes[machine]['type'] = 'machine'
            self.G.nodes[source]['type'] = 'source'

    def compute_density_field(self, pairs, lambda_factor=2.0):
        """
        pairs: liste de tuples (noeud_machine, noeud_source, aire_section)
        lambda_factor: contrôle à quelle vitesse la gravité s'estompe avec la distance.
        """
        # 1. On récupère toutes les arêtes et on calcule leurs milieux
        edges = list(self.G.edges())
        midpoints = np.array([(self.get_pos(u) + self.get_pos(v)) / 2.0 for u, v in edges])

        # Tableau de densité initialisé à 0 pour chaque arête
        densities = np.zeros(len(edges))

        # 2. On additionne les champs de gravité de chaque ligne de désir
        for machine, source, area in pairs:
            A = self.get_pos(machine)
            B = self.get_pos(source)

            # Calcul magique vectorisé des distances
            dists = self.point_to_segment_dist_vectorized(midpoints, A, B)

            # Application de la force : A * exp(-lambda * d^2)
            force = area * np.exp(-lambda_factor * (dists**2))
            densities += force

        # 3. On enregistre cette densité comme un attribut dans le graphe
        for i, (u, v) in enumerate(edges):
            self.G[u][v]['density'] = densities[i]

            # EXEMPLE DE DISTORSION DU COÛT : 
            # Plus la densité est forte, plus le "coût routé" baisse (pour attirer Dijkstra)
            base_dist = self.G[u][v]['weight']
            self.G[u][v]['routed_cost'] = base_dist / (1 + densities[i])

    def plot_density(self):
        """Affiche le graphe avec une heatmap sur les arêtes selon leur densité."""
        pos = nx.get_node_attributes(self.G, 'pos')
        edges = self.G.edges()

        # On récupère les densités pour colorer les arêtes
        densities = [self.G[u][v].get('density', 0) for u, v in edges]

        plt.figure(figsize=(10, 8))

        # Dessin des arêtes avec une colormap (cmap) allant du gris au rouge vif
        edges_draw = nx.draw_networkx_edges(
            self.G, pos, edge_color=densities, 
            edge_cmap=plt.cm.Reds, width=3, alpha=0.8
        )

        nx.draw_networkx_nodes(self.G, pos, node_size=20, node_color='black')

        # Barre de légende pour la chaleur
        plt.colorbar(edges_draw, label='Densité (Champ de Gravité)')
        plt.title("Carte des densités des câbles (Lignes de désir)")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()

    def plot(self, save_path=None, show=True):
        pos = nx.get_node_attributes(self.G, 'pos')

        # On sépare les noeuds par catégorie
        machines = [n for n in self.G.nodes if self.G.nodes[n].get('type') == 'machine']
        sources = [n for n in self.G.nodes if self.G.nodes[n].get('type') == 'source']
        other_initials = [n for n in self.G.nodes if self.G.nodes[n].get('is_initial') and not self.G.nodes[n].get('type')]
        added_nodes = [n for n in self.G.nodes if not self.G.nodes[n].get('is_initial')]

        plt.figure(figsize=(10, 10))

        # Dessin du réseau (arêtes et noeuds intermédiaires)
        nx.draw_networkx_edges(self.G, pos, edge_color='black', alpha=0.3)
        if added_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=added_nodes, node_size=20, node_color='gray')

        # Dessin des points de base
        if other_initials:
            nx.draw_networkx_nodes(self.G, pos, nodelist=other_initials, node_size=80, node_color='green', alpha=0.5)
        if machines:
            nx.draw_networkx_nodes(self.G, pos, nodelist=machines, node_size=200, node_color='magenta', node_shape='s', label='Machines')
        if sources:
            nx.draw_networkx_nodes(self.G, pos, nodelist=sources, node_size=300, node_color='orange', node_shape='*', label='Sources')

        # Tracé des lignes de désir
        if hasattr(self, 'demand_pairs'):
            for m, s, energy in self.demand_pairs:
                plt.plot([pos[m][0], pos[s][0]], [pos[m][1], pos[s][1]],
                         color='purple', linestyle=':', alpha=0.4, linewidth=energy/2)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc="upper right")
        plt.title("Réseau Physique et Lignes de Désir")

        # GESTION DE LA SAUVEGARDE ET DE L'AFFICHAGE
        if save_path:
            # bbox_inches='tight' évite que les bords de l'image soient coupés
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()  # Libère la mémoire si on ne fait que sauvegarder

    def save(self, graph_name, base_dir="tests_2/graphs"):
        """
        Crée un sous-dossier graph_name et y sauvegarde le pickle et le plot.
        """
        # Création du chemin vers le sous-dossier (ex: test_2/graphs/mon_reseau_1)
        target_dir = Path(base_dir) / graph_name

        # Création du dossier et de ses parents si nécessaire
        target_dir.mkdir(parents=True, exist_ok=True)

        pkl_path = target_dir / f"{graph_name}.pkl"
        png_path = target_dir / f"{graph_name}.png"

        # 1. Sauvegarde visuelle (sans bloquer l'exécution avec show=False)
        self.plot(save_path=png_path, show=False)

        # 2. Sauvegarde de l'objet Python
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"💾 Réseau '{graph_name}' sauvegardé avec succès dans : {target_dir}")

    @classmethod
    def load(cls, graph_name, base_dir="tests_2/graphs"):
        """
        Recharge une instance depuis son sous-dossier.
        """
        pkl_path = Path(base_dir) / graph_name / f"{graph_name}.pkl"

        try:
            with open(pkl_path, 'rb') as f:
                network = pickle.load(f)
            print(f"📂 Réseau '{graph_name}' chargé avec succès depuis : {pkl_path}")
            return network
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{pkl_path}' est introuvable.")
            return None
# # Execution
# net = RobustNetwork(n_nodes = 15, n_segments=30)
# # 1. Génération
# net.generate()

# # 2. CORRECTION DES SUPERPOSITIONS (Ton intuition était la bonne)
# net.resolve_colinear_overlaps()

# # 3. Nettoyage des noeuds inutiles alignés
# net.simplify_collinear_nodes()

# # 4. Découpage pour tes supports physiques
# net.subdivide_long_edges()

# # 5. Choix des paires machines/sources
# net.generate_demand_pairs()

# net.plot()

# #python -m tests_2.graph
