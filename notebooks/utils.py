import os
import re
import torch

def find_my_synapses(source_directory, data_name):
    pattern = re.compile(rf"^{data_name}.*_(\d+)hu\.pt$")
    hidden_units_files = {}

    for file in os.listdir(source_directory):
        if file.endswith('.pt') and file.startswith(data_name):
            match = pattern.match(file)
            if match:
                hidden_units = int(match.group(1))
                hidden_units_files[hidden_units] = os.path.join(source_directory, file)
    return hidden_units_files

