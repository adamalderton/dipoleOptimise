import sys, os
import math
import json

import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.animation as ani

import matplotlib as mpl
mpl.rcParams["animation.ffmpeg_path"] = "C:\\FFmpeg\\bin\\ffmpeg"

################################################
os.chdir("..\\..")

################################################

    
def display_result(config):
    
    fig = plt.figure(figsize = ((6, 8)))
    
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1, rowspan = 2)
    
    # Load dipoles
    dipole_pos = np.genfromtxt("experiments\\{}\\final_dipole_placement.txt".format(config["experiment_name"]))
    
    def animate(input_num):
        ax1.clear()
        ax2.clear()

        focus_point = config["source_dipoles"][input_num]["focus_point"]
        
        #******* Detector Plot *******#
        
        intensity_data = np.genfromtxt("experiments//{}//result_fields//input_{}_detector_field.txt".format(config["experiment_name"], input_num))
        ax1.plot(intensity_data, "k-", label = "Detector Field")
        
        ax1.set_xlabel("X")
        ax1.set_ylabel("|E|")
        ax1.set_title("Input = {}, FP = ({}, {}), |E| = {}".format(input_num, focus_point[0], focus_point[1], intensity_data[focus_point[0]]))
        
        #******* Dipoles Plot *******#
        
        ax2.scatter(dipole_pos[:,1], dipole_pos[:,0], c = "yellow")
        
        field_data = np.genfromtxt("experiments//{}//result_fields//input_{}_lattice_field.txt".format(config["experiment_name"], input_num))
        ax2.imshow(field_data.T, interpolation = "bicubic", cmap = "Greys", vmin = -1, vmax = 1, extent = [0, config["scatterer_dimensions"][1] + config["extra_z_evaluation"], 0, config["scatterer_dimensions"][0]], origin = "lower")
        
        ax2.set_xlim(0, config["scatterer_dimensions"][1] + config["extra_z_evaluation"])
        ax2.set_ylim(0, config["scatterer_dimensions"][0])
        ax2.set_aspect('equal', 'box')
        ax2.set_title("Dipole Positions")
        ax2.invert_yaxis()
        ax2.set_xlabel("Z, Propagation ->")
        ax2.set_ylabel("X")
        
        #******* Detector line on dipoles plot ********#
        ax2.axvline(x = config["scatterer_dimensions"][1], color = "#FFA500", linewidth = 3)
        
        #******* Scatterer boundary line **************#
        ax2.axvline(x = config["detector_z"], color = "#AFDEEE", linewidth = 3)

        # Add focus point patch
        ax2.add_patch(patches.Circle([focus_point[1], focus_point[0]], radius = 20, color="orange"))

        plt.tight_layout()
        
    animation = ani.FuncAnimation(fig, animate, config["num_source_dipoles"], interval = 2 * 1000)
    
    writervideo = ani.FFMpegWriter(fps = 1)
    animation.save("experiments\\{}\\results.mp4".format(config["experiment_name"]), writervideo)

def main(experiment_name):
    with open("experiments\\" + experiment_name + "\\experiment_config.json", "r") as jf:
        config = json.load(jf)
    
    display_result(config)

if __name__ == "__main__":
    main(sys.argv[1])
