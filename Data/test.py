import pickle
import os
from pathlib import Path

# Open the file in 'rb' (read binary) mode
if __name__ == "__main__":
    print("ok")
    with open("C:/Users/bapti/Documents/ENSAE et master/Cours Master/Capstone/Capstone_M2DS_26/Data/graph_data_exemple.pkl", "rb") as file:
        obj = pickle.load(file)
