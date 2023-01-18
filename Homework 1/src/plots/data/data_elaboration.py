"""
You can write here the data elaboration function/s
You should read all the JSON files containing simulations results and compute
average and std of all the metrics of interest.
You can find the JSON file from the simulations into the data.evaluation_tests folder.
Each JSON file follows the naming convention: simulation-current date-simulation id__seed_drones number_routing algorithm
In this way you can parse the name and properly aggregate the data.
To aggregate data you can use also external libraries such as Pandas!
IMPORTANT: Both averages and stds must be computed over different seeds for the same metric!
"""
import os
import json
import pandas as pd
import numpy as np
from itertools import product

def compute_data_avg_std(path: str):
    """
    Computes averages and stds from JSON files
    @param path: results folder path
    @return: one or more data structure containing data
    """
    data = pd.DataFrame(data_aggregator(path))
    output = []
    algs = data[0].unique()
    n_drones = data[1].unique()
    for t in list(product(algs, n_drones)):
        t_data = data[(data[0] == t[0]) & (data[1] == t[1])]
        avg_ptd = np.average(t_data[2])
        avg_pdt = np.average(t_data[3])
        avg_mr = np.average(t_data[4])
        std_ptd = np.std(t_data[2])
        std_pdt = np.std(t_data[3])
        std_mr = np.std(t_data[4])
        output.append([t[0], t[1], avg_ptd, avg_pdt, avg_mr, std_ptd, std_pdt, std_mr])
    return pd.DataFrame(output, columns = ['alg', 'ndrones', 'avg_ptd', 'avg_pdt', 'avg_mr', 'std_ptd', 'std_pdt', 'std_mr'])

def data_aggregator(path):
    data = []
    fpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', path))
    for file in os.listdir(fpath):
        file = json.loads(open(os.path.join(fpath, file), "r").read())
        nd = file["mission_setup"]["n_drones"]
        alg = file["mission_setup"]["routing_algorithm"].split(".")[1]
        ptd = file["number_of_packets_to_depot"]
        pdt = file["packet_mean_delivery_time"]
        mr = file["mean_number_of_relays"]
        data.append(tuple((alg, nd, ptd, pdt, mr)))
    return data

if __name__ == "__main__":
    """
    You can run this file to test your script
    """

    path = "data/evaluation_tests"

    compute_data_avg_std(path=path)