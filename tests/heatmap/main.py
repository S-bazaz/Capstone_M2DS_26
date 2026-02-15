from classes.graph_structure import DynamicGraph
from classes.discretisation import DiscretizedSpace
from classes.geometry import Point, Points, Segment, Segments
from pathlib import Path

#python -m heatmap.main

### ========> Begin Parameters <======== ###
x_size = 10
y_size = 10
n_segments = 35
n_node = 20
discretisation = 100
discretisation_2 = 50

name = "05"
mu_list = [1/2, 1, 2, 5, 10]
### ========> End Parameters <======== ###





G = DynamicGraph(x_size = x_size, y_size = y_size,
                n_segments= n_segments, n_node= n_node,
                name = name, n_machine=5,
                )

G.save()

G: DynamicGraph = DynamicGraph.load(name)


def distance(x1, y1, x2, y2, mu):
    d = ((x1-x2)*(x1 - x2) + (y1-y2)*(y1 - y2))**(1/2)
    a = max(1 - d/mu, 0) 
    return a

def distance2(x1, y1, x2, y2, mu):
    d = ((x1-x2)*(x1 - x2) + (y1-y2)*(y1 - y2))**(1/2)
    a = 1/(1+mu*d)
    return a


space_1 = DiscretizedSpace(G, subdivision = discretisation, name = "space1")
for coupling in G.coupling:
    subdivision_per_unit = 10
    w = coupling.w
    p1 = coupling.energy
    p2 = coupling.machine
    d = Points.calculate_distance(p1, p2)

    subdivision = int(d*subdivision_per_unit)

    v = p1.get_direction(p2)

    v_x, v_y = v[0] / subdivision, v[1] / subdivision
    x, y = p1.x, p1.y
 
    space_1.update_case(x, y, w)
    for i in range(int(d*subdivision_per_unit)):
        x, y = x + v_x, y+ v_y
        space_1.update_case(x, y, w)
space_1.plot(show = False, save = True)



for mu in mu_list:
    print(mu)
    name = "mu_" + str(mu)
    print(name)

    space_2 = DiscretizedSpace(G, subdivision = discretisation_2, name = name)

    target_points = []
    step_2 = x_size / discretisation_2
    for a in range(discretisation_2):
        for b in range(discretisation_2):
            cx = (a + 0.5) * step_2
            cy = (b + 0.5) * step_2
            target_points.append((a, b, cx, cy))

    step_1 = x_size / discretisation
    for i in range(discretisation):
        sx = (i + 0.5) * step_1
        for j in range(discretisation):
            sy = (j + 0.5) * step_1
            
            val_1 = space_1.case_value(sx, sy)
            if val_1 == 0: continue
            for a_idx, b_idx, tx, ty in target_points:
                influence = distance2(sx, sy, tx, ty, mu) * val_1
                space_2.update_case(tx, ty, influence)
            

    space_2.plot(show = False, save = True)