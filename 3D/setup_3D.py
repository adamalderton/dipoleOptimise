import os, json, random
import numpy as np

os.chdir("..//..")

def sanity_check(experiment_config):
    if experiment_config["placement_algorithm"] not in globals():
        return False, "Not a valid placement algorithm."
    
    if experiment_config["placement_algorithm"] in ["main", "sanity_check"]:
        return False, "Not a valid placement algorithm."
    
    if experiment_config["dimensions"] not in [2, 3]:
        return False, "Dimensionality must be 2 or 3."
    
    if (isinstance(experiment_config["save_positions_every_X_epochs"], int) == False):
        return False, "Frames/epochs must be an integer"
    
    if len(experiment_config["scatterer_dimensions"]) != experiment_config["dimensions"]:
        return False, "Number of scatterer dimensions must match dimensionality."
    
    for i, length in enumerate(experiment_config["scatterer_dimensions"]):
        if not (isinstance(length, float) or isinstance(length, int)):
            return False, "Invalid scatterer dimension, dimension {}.".format(["X", "Y", "Z"][i])
        if length <= 0.0:
            return False, "Invalid scatterer dimension, dimension {}.".format(["X", "Y", "Z"][i])
        if length % 2 != 0:
            return False, "Scatterer dimensions must be a multiple of 2."
    
    if not isinstance(experiment_config["num_dipoles"], int):
        return False, "Number of dipoles must be an integer."
    
    if experiment_config["num_dipoles"] <= 0:
        return False, "Number of dipoles needs to be positive and non-zero."
    
    if experiment_config["polarisability"] < 0:
        return False, "Dipole polarisabilty cannot be negative."
    
    if experiment_config["resolution_factor"] % 2 != 0:
        return False, "Resolution factor must be a power of 2."
    
    if experiment_config["resolution_factor"] <= 0:
        return False, "Resolution factor must be positive." 
    
    return True, None

def random_placement(experiment_config):
    with open("experiments\\{}\\initial_dipole_placement.txt".format(experiment_config["experiment_name"]), "a") as f:
        zeroes_pad = len(str(max(experiment_config["scatterer_dimensions"])))    # Ensure numbers generated are all same length, in string format
        
        for _ in range(experiment_config["num_dipoles"]):
            for i in range(experiment_config["dimensions"]):
                f.write("{}\t".format(str(random.randint(0, experiment_config["scatterer_dimensions"][i])).zfill(zeroes_pad), end = ""))
            f.write("\n")

def main():
    # Create directory to store experiment files and data
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    # Create experiment directory
    while True:
        experiment_name = str(input("\tPlease enter a name for the experiment:\n"))
        if os.path.isdir("experiments\\" + experiment_name):
            print("\tAn experiment directory with this name already exists. Please try again.")
        else:
            os.mkdir("experiments\\" + experiment_name)
            os.mkdir("experiments\\" + experiment_name + "\\epochs")
            os.mkdir("experiments\\" + experiment_name + "\\result_fields")
            break
        
    default_values = {
        # Default values for general setup
        "experiment_name" : experiment_name,
        "complete" : False,
        "dimensions" : 3,
        "num_dipoles" : 50,
        "polarisability" : 5.0,
        "wavelength" : 300,
        "num_source_dipoles" : 1,
        "scatterer_dimensions" : [1024, 1024, 1024],
        "detector_z" : 1200,
        "extra_z_evaluation" : 384,
        "placement_algorithm" : "random_placement",
        "max_iterations" : 200,
        "save_positions_every_X_epochs" : 0, # Set to zero to not save any positions apart from final. Set to 1 to save every epoch.
        "resolution_factor" : 16     # Factor by which to reduce field evaluations and interpolate between. Must be power of 2.
    }

    # Load default values into json file
    with open("experiments\\" + experiment_name + "\\experiment_config.json", "w") as jf:
        jf.write(json.dumps(default_values, indent = 4))
        
    print("\tDirectory \"{}\" and config file have been created.".format(experiment_name))
    print("\tPlease review the config, save it, and press enter here to continue. (Ctrl+C to quit)")
    os.system("notepad.exe experiments\\{}\\experiment_config.json".format(experiment_name))
    input("")
    
    print("\tSetting up...")
    
    # Re-load experiment config, with updated user changes.
    # Also check the validity of the input files and parameters within it.
    with open("experiments\\" + experiment_name + "\\experiment_config.json", "r") as jf:
        experiment_config = json.load(jf)    # Attempt to load experiment parameters
        sane, message = sanity_check(experiment_config)    # Check parameters
        if not sane:
            print("\t{}".format(message))
            print("\tPlease delete the experiment directory to try again.")
        
    ######## WRITE CODE HERE TO SETUP SOURCE DIPOLES #######
    experiment_config["source_dipoles"] = [
        {
            "position" : [512, 512, -4096],
            "focus_point" : [512, 512, 1200],
            "mod_polarisation" : 1.0
        },
        {
            "position" : [512, 512 - 1024, -4096],
            "focus_point" : [512, 768, 1200],
            "mod_polarisation" : 1.0
        },
        {
            "position" : [512, 512 + 1024, -4096],
            "focus_point" : [512, 256, 1200],
            "mod_polarisation" : 1.0
        }
    ]
    experiment_config["num_source_dipoles"] = len(experiment_config["source_dipoles"])
    
    # Write final confif into file
    with open("experiments\\" + experiment_name + "\\experiment_config.json", "w") as jf:
        jf.write(json.dumps(experiment_config, indent = 4))
    
    # Run the decided placement algorithm (it was checked to be valid above)
    globals()[experiment_config["placement_algorithm"]](experiment_config)
        
    print("\n\tSetup complete!\n")
    print("\tPlease now run your experiment with the command ./engine.exe [name].")

if __name__ == "__main__":
    main()
