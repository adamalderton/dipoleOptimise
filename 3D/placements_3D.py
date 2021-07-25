import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

os.chdir("..\\..")

def main(config, view):
    fig = plt.figure(figsize=(6, 6))
    
    data = np.genfromtxt("experiments\\{}\\final_dipole_placement.txt".format(config["experiment_name"]))

    if view == "3D":
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(data[:,2], data[:,0], data[:,1])

        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")
        
    elif view == "XY":
        ax = fig.add_subplot()
        
        ax.scatter(data[:,0], data[:,1])
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        ax.set_xlim(0, config["scatterer_dimensions"][0])
        ax.set_ylim(0, config["scatterer_dimensions"][1])
        ax.set_aspect("equal")
    
    elif view == "ZX":

        ax = fig.add_subplot()
        
        ax.scatter(data[:,2], data[:,0])
        
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        
        ax.set_xlim(0, config["scatterer_dimensions"][2])
        ax.set_ylim(0, config["scatterer_dimensions"][0])
        ax.set_aspect("equal")
    
    elif view == "ZY":
        ax = fig.add_subplot()
        
        ax.scatter(data[:,2], data[:,1])
        
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")

        ax.set_xlim(0, config["scatterer_dimensions"][2])
        ax.set_ylim(0, config["scatterer_dimensions"][1])
        ax.set_aspect("equal")
        
    else:
        raise Exception("View invalid. Valid views: 3D, XY, ZX, ZY.")

    plt.show()

if __name__ == "__main__":
    if (len(sys.argv)) <= 2:
        raise Exception("Paramters not correctly passed.")
    
    experiment_name = sys.argv[1]
    view = sys.argv[2]

    with open("experiments\\" + experiment_name + "\\experiment_config.json", "r") as jf:
        config = json.load(jf)
    
    main(config, view)