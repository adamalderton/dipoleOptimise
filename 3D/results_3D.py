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
os.chdir("..//..")

################################################

    
def display_multiple(config):
    
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot()
    
    def animate(input_num):
        ax.clear()

        focus_point = config["source_dipoles"][input_num]["focus_point"]
        
        intensity_data = np.genfromtxt("experiments//{}//result_fields//input_{}_detector_field.txt".format(config["experiment_name"], input_num))
        
        ax.imshow(intensity_data,  interpolation = "bicubic", cmap = "gist_gray")
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Z = {}, input_num = {}, FP = ({}, {}, {}), |E| = {}".format(config["detector_z"], input_num, focus_point[0], focus_point[1], focus_point[2], intensity_data[focus_point[0]][focus_point[1]]))
        
        # Add focus point patch
        ax.add_patch(patches.Circle([focus_point[1], focus_point[0]], radius = config["resolution_factor"]/3, color="red"))

        fig.tight_layout()
        
    animation = ani.FuncAnimation(fig, animate, config["num_source_dipoles"], interval = 2 * 1000)
    
    writervideo = ani.FFMpegWriter(fps = 1)
    animation.save("experiments\\{}\\results.mp4".format(config["experiment_name"]), writervideo)

def display_single(config):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot()

    focus_point = config["source_dipoles"][0]["focus_point"]
        
    intensity_data = np.genfromtxt("experiments//{}//result_fields//input_{}_detector_field.txt".format(config["experiment_name"], 0))
        
    ax.imshow(intensity_data.T,  interpolation = "bicubic", cmap = "gist_gray")
        
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Z = {}, FP = ({}, {}, {}), |E| = {}".format(config["detector_z"], focus_point[0], focus_point[1], focus_point[2], intensity_data[focus_point[0]][focus_point[1]]))
        
    # Add focus point patch
    ax.add_patch(patches.Circle([focus_point[1], focus_point[0]], radius = config["resolution_factor"]/3, color="red"))

    fig.tight_layout()

    fig.savefig("experiments\\{}\\results.png".format(config["experiment_name"]))
    
def main(experiment_name):
    with open("experiments\\" + experiment_name + "\\experiment_config.json", "r") as jf:
        config = json.load(jf)
        
    if config["num_source_dipoles"] == 0:
        raise Exception("num_inputs is 0.")
    elif config["num_source_dipoles"] == 1:
        display_single(config)
    else:
        display_multiple(config)

if __name__ == "__main__":
    main(sys.argv[1])