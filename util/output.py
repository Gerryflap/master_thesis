import datetime
import os
import argparse
import json


def get_output_path(dataset_name, experiment_name):
    """
    Gets the output path where experiment results can be put. Only call this once since it does include time/date
    :param dataset_name: name of the dataset used (eg. mnist/celeba/celeba_cropped)
    :param experiment_name: name of the individual experiment (eg. gan/vae/morgan/...)
    :return: The output path
    """
    date = datetime.datetime.now().replace(microsecond=0).isoformat()
    path = os.path.join(".", "results", dataset_name, experiment_name, date)
    return path


def path_to_folder_names(path):
    (head, tail) = os.path.split(path)
    if head == "":
        return []
    previous_paths = path_to_folder_names(head)
    if tail == "":
        return previous_paths
    else:
        previous_paths.append(tail)
    return previous_paths


def make_result_dirs(path):
    splitted = path_to_folder_names(path)[1:]
    current_path = "./results"
    if not os.path.isdir(current_path):
        raise IOError("Cannot create results folder since the script is not executed from the root project folder. "
                      "Try running it from the root folder with python -m and "
                      "check whether the folder results exists in the root folder!")
    for dir in splitted:
        new_path = os.path.join(current_path, dir)
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        current_path = new_path


def save_hyperparameters(folder_path, args: argparse.Namespace):
    arg_dict = vars(args)

    with open(os.path.join(folder_path, "hyperparams.json"), "w") as f:
        json.dump(arg_dict, f)


def init_experiment_output_dir(dataset_name: str, experiment_name: str, args: argparse.Namespace):
    # Generate the output path string
    path = get_output_path(dataset_name, experiment_name)

    # Create any missing directories
    make_result_dirs(path)

    # Save experiment command line arguments
    save_hyperparameters(path, args)
    return path