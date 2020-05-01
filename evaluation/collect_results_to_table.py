"""
    Collects all results from the output directory of the mmpmr_evaluator and puts them in a
    format that's easy to paste into a spreadsheet.
"""

import json
import os

base_path = "results/celeba64/model_evaluation"

folders = sorted(os.listdir(base_path))

print("path\tmmpmr\treconstruction rate\tmean distance from morph to x1 and x2\tmean distance from reconstruction to x")
for folder in folders:
    path = os.path.join(base_path, folder, "results.json")
    with open(path, "r") as f:
        data = json.load(f)

        s = data['path'] + "\t"
        s += "%f\t" % data['mmpmr']
        s += "%f\t" % data['rr']
        s += "%f\t" % data['mmd']
        s += "%f\t" % data['mrd']
        print(s)
