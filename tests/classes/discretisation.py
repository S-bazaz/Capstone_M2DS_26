from .geometry import *
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from classes.graph_structure import DynamicGraph

class DiscretizedSpace:
    def __init__(self, graph: DynamicGraph, subdivision = 100, name = None):
        self.subdivision = subdivision
        self.x_size = graph.x_size
        self.y_size = graph.y_size
        self.x_step = graph.x_size/subdivision
        self.y_step = graph.y_size/subdivision
        self.name = name
        self.path = graph.path
        self.matrix = np.zeros((subdivision, subdivision))
    
    def get_path(self):
        prefix = self.path
        if self.name is not None:
            name = self.name + ".png"
            return prefix / name
        else:
            return prefix / "un-named"

    def get_case(self, x, y):
        """
        Take as input spatial coordinate and return the corresponding index in the array
        Note that this is "inverted" as point (0,0) correspond to the last lign first column.
        All the access to the table will go through this to avoid human errors in indexes.
        """
        if x < 0 or x > self.x_size or y < 0 or y > self.y_size:
            raise ValueError("Out of bounds")
        else:
            j = min(int(x // self.x_step), self.subdivision - 1)
            i = min(int((self.y_size - y) // self.y_step), self.subdivision - 1)
            return (i,j)

    def case_value(self, x, y):
        i, j = self.get_case(x, y)
        return self.matrix[i,j] 

    def update_case(self, x, y, value):
        i, j = self.get_case(x, y)
        self.matrix[i,j] += value

    def plot(self, title="Spatial Distribution", cmap="YlOrRd", show = True, save = False):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.matrix, cmap=cmap, cbar_kws={'label': 'Intensity'})
        plt.title(title)
        plt.xlabel("X Coordinate (Discretized)")
        plt.ylabel("Y Coordinate (Discretized)")
        if show:
            plt.show()
        if save:
            plt.savefig(self.get_path())
        plt.close()
    
