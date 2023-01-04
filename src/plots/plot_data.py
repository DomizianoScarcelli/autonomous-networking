"""
Implement here your plotting functions
Below you can see a print function example.
You should use it as a reference to implements your own plotting function
IMPORTANT: if you need you can and should use other matplotlib functionalities! Use
            the following example only as a reference
The plot workflow is can be summarized as follows:
    1) Extensive simulations
    2) Json file containing results
    3) Compute averages and stds for each metric for each algorithm
    4) Plot the results
In order to maintain the code tidy you can use:
    - src.plots.config.py file to store all the parameters you need to
        get wonderful plots (see the file for an example)
    - src.plots.data.data_elaboration.py file to write the functions that compute averages and stds from json
        result files
    - src.plots.plot_data.py file to make the plots.
The script plot_data.py can be run using python -m src.plots.plot_data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from src.experiments.json_and_plot import ALL_SIZE
from src.plots.config import PLOT_DICT, LABEL_SIZE, LEGEND_SIZE, METRIC_NAME, TICKS_SIZE, OTHER_SIZES
from src.plots.data import data_elaboration

def plot(algorithms: list,
         y_data: list,
         y_data_std: list,
         type: str):

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(13, 9))
    print(f"Algorithms: {algorithms}")
    print(f"y_data: {y_data}\ny_data_std: {y_data_std}")
    print(PLOT_DICT[algorithms[0]]["x_ticks_positions"])
    
    title = ""
    for i, alg in enumerate(algorithms):
        ax1.errorbar(x=np.array(PLOT_DICT[alg]["x_ticks_positions"]), y=y_data[i], yerr=y_data_std[i], color=PLOT_DICT[alg]["color"])
        title += alg + "=" + PLOT_DICT[alg]["full_name"] + "; "

    ax1.set_title(title, fontsize=OTHER_SIZES)
    ax1.set_ylabel(ylabel=METRIC_NAME[type], fontsize=LABEL_SIZE)
    ax1.set_xlabel(xlabel="Number of Drones", fontsize=LABEL_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICKS_SIZE)

    leg_loc = ""
    if type == "PDT":
        leg_loc = "upper right"
    else:
        leg_loc = "upper left"

    plt.legend([a for a in algorithms], ncol=len(algorithms),
               handletextpad=0.1,
               columnspacing=0.7,
               prop={'size': LEGEND_SIZE},
               loc=leg_loc)

    plt.grid(linewidth=0.3)
    plt.tight_layout()
    plt.savefig("src/plots/figures/" + type + ".svg")
    plt.savefig("src/plots/figures/" + type + ".png", dpi=400)
    plt.clf()

if __name__ == "__main__":
    """
    Run this file to get the plots.
    Of course, since you need to plot more than a single data series (one for each algorithm) you need to modify
    plot() in a way that it can handle a multi-dimensional data (one data series for each algorithm). 
    y_data and y_data_std could be for example a list of lists o a dictionary containing lists. It up to you to decide
    how to deal with data
    """

    data = pd.DataFrame(data_elaboration.compute_data_avg_std("data/evaluation_tests"))
    data = data.sort_values(by = "ndrones")

    algorithms = ["RND"]
    metrics = {"PTD": ["avg_ptd, std_ptd"], "PDT": ["avg_pdt, std_pdt"], "MR": ["avg_mr, std_mr"]}
    for metric in metrics.keys():
        avg_col_name = "avg_"+metric.lower()
        std_col_name = "std_"+metric.lower()
        y_data = [data.iloc[np.where(data["alg"]==alg)][avg_col_name].values for alg in algorithms]
        y_data_std = [data.iloc[np.where(data["alg"]==alg)][std_col_name].values for alg in algorithms]
        type = metric
        plot(algorithms=algorithms, y_data=y_data, y_data_std=y_data_std, type=type)