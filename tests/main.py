from graph_class import *
from generation_scripts import generate_and_save_graphs

x_size = 10
y_size = 10
inv = lambda x: 1/(1E-6 + x)
sqrt_inv = lambda x: 1/np.sqrt(1E-6 + x)
affine = lambda x: (1 - x/np.sqrt(x_size**2 + y_size**2))


graph = DynamicGraph(size_x=10, size_y=10, n_node=15, n_segments=20, f=inv, auto_gen=False)
graph.generate_live(delay=2)

# generate_and_save_graphs(
#     node_steps=[10, 20, 50], 
#     segment_steps=[15, 30, 60], 
#     n_variants=10
# )