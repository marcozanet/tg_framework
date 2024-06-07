import os, sys, shutil
from glob import glob 
import platform
import random
from pathlib import Path
from PIL import Image
import yaml 
from yaml import SafeLoader
import tqdm
import yaml
from typing import Any


def setup_paths() -> tuple:
    """ SETS UP PATHS TO YAML DEPENDING ON THE SYSTEM USED. """

    if platform.system() == 'Windows':
        safe_copy_yaml = 'testing_safe_windows.yaml'
        yaml_fp = 'testing_windows.yaml'

    elif platform.system() == 'Darwin':
        safe_copy_yaml = 'testing_safe_mac.yaml'
        yaml_fp = 'testing_mac.yaml'
    else:
        raise NotImplementedError("The operating system is:", platform.system())

    return  safe_copy_yaml, yaml_fp


def restore_yaml(original_yaml:str, modified_yaml) -> None:
    """ OVERWRITES THE MODIFIED FILE WITH THE ORIFINAL FILE """

    with open(original_yaml, 'r') as file:
        data = yaml.safe_load(file)
    with open(modified_yaml, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

    return


def get_last_modified_folder(parent_folder) -> Path:
    """ GETS THE LAST MODIFIED SUBFOLDER FROM A FOLDER (to get the last model output dir) """

    # List all subdirectories in the parent folder
    subdirs = [d for d in glob(os.path.join(parent_folder, '*/')) if os.path.isdir(d)]
    if not subdirs:
        return None  # No subdirectories found
    # Get the creation times of all subdirectories
    subdirs_with_ctime = [(d, os.path.getctime(d)) for d in subdirs]
    # Sort subdirectories by creation time, in descending order
    subdirs_with_ctime.sort(key=lambda x: x[1], reverse=True)
    # Return the path of the most recently created subdirectory

    return subdirs_with_ctime[0][0]


def is_there_incoherence_in_params(yaml_fp:str) -> bool:
    """ PROVIDES TRUE IF THERE'S A WRONG COMBINATION OF PARAMS WHICH MAKES NO SENSE
        E.g., task = 'segmentation', but  'trained_model_weights' are weight from a detection model """

    with open(yaml_fp, 'r') as file:
        data = yaml.safe_load(file)
    dictionary =  data['inference']
    if dictionary['task']=='segmentation' and ('segm' not in dictionary['trained_model_weights']): return True
    elif dictionary['task']=='detection' and ('detect' not in dictionary['trained_model_weights']): return True
    elif dictionary['create_grayscale_masks'] is True and dictionary['save_txt'] is False: return True
    else: return False


def edit_yaml_fp(yaml_fp:str, field:str, var:str, new_val:Any, verbose:bool=False) -> Any:
    """ MODIFIES YAML FILE BY CHANGING A VARIABLE IN A FIELD WITH A NEW VAL"""

    with open(yaml_fp, 'r') as file:
        data = yaml.safe_load(file)
    if verbose is True:
        print("Original data:", data)
        print("Original data:", data[field][var])
    old_val = data[field][var]
    data[field][var] = new_val
    if verbose is True: print("Modified data:", data[field][var])
    with open(yaml_fp, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)
    
    return old_val


def remove_last_item_yaml_fp(yaml_fp:str, verbose:bool=False) -> dict:
    """ REMOVES THE LAST VARIABLE FROM A YAML FILE """

    with open(yaml_fp, 'r') as file:
        data = yaml.safe_load(file)
    dictionary = data['inference']
    if verbose is True: print(f"Data before: {data}")
    rem_item = dictionary.popitem()
    if verbose is True: print(f"Removed item: {rem_item}")
    data = {'inference': dictionary}
    if verbose is True: print(f"Data after: {data}")
    with open(yaml_fp, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

    return {rem_item[0]: rem_item[1]}


def add_item_yaml_fp(yaml_fp:str, item: dict, verbose:bool = False) -> None:
    """ ADDS A VARIABLE TO THE YAML FILE """

    with open(yaml_fp, 'r') as file:
        data = yaml.safe_load(file)
    if verbose is True: print(f"Old dictionary: {data}")
    dictionary = data['inference']
    dictionary.update(item)
    if verbose is True: print(f"Added item: {item}")
    data = {'inference': dictionary}
    if verbose is True: print(f"New dictionary: {data}")
    with open(yaml_fp, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)
    
    return


def find_pt_files(home_dir:str) -> list:
    """ HELP FUNC TO FIND ALL MODEL WEIGHTS IN .pt FORMAT """

    pt_files = []
    for root, _, files in os.walk(home_dir):
        for file in files:
            if file.lower().endswith('.pt'):
                pt_files.append(os.path.join(root, file))

    return pt_files


def get_config_params(yaml_fp:str, config_name:str) -> dict:
    """ GETS PARAMS CONTAINED IN A .YAML FILE """

    with open(yaml_fp, 'r') as f: 
        all_params = yaml.load(f, Loader=SafeLoader)
    params = all_params[config_name]
    
    return  params





if __name__ == "__main__": 
    last = get_last_modified_folder('/Users/marco/tg/tg_framework/testing_output')
    print(last)