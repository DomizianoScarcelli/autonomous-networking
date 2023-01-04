import numpy as np
import matplotlib.pyplot as plt

LABEL_SIZE = 32
LEGEND_SIZE = 30
TITLE_SIZE = 36
TICKS_SIZE = 20
OTHER_SIZES = 20

METRICS_OF_INTEREST = [
    "number_of_packets_to_depot",
    "packet_mean_delivery_time",
    "mean_number_of_relays"]

METRIC_NAME = {
    "PTD": "Number Of Packets To Depot",
    "PDT": "Packet Mean Delivery Time",
    "MR": "Mean Number Of Relays"
}

SCALE_LIM_DICT = {
    "number_of_packets_to_depot": {
        "scale": "linear",
        "ylim": (0, 1000)
    },
    "packet_mean_delivery_time": {
        "scale": "linear",
        "ylim": (0, 5)
    },
    "mean_number_of_relays": {
        "scale": "linear",
        "ylim": (0, 10)
    }
}

PLOT_DICT = {
    "RND": {
        "hatch": "",
        "markers": "X",
        "linestyle": "-",
        "color": 'tab:blue',
        "label": "RND",
        "x_ticks_positions": np.array(np.arange(5, 35, 5)),
        "full_name": "RandomRouting"
    },
    "GEO": {
        "hatch": "",
        "markers": "p",
        "linestyle": "-",
        "color": 'tab:orange',
        "label": "GEO",
        "x_ticks_positions": np.array(np.arange(5, 35, 5)),
        "full_name": "GeoRouting"
    },
    "SQL": {
        "hatch": "",
        "markers": "s",
        "linestyle": "-",
        "color": 'tab:green',
        "label": "SQL",
        "x_ticks_positions": np.array(np.arange(5, 35, 5)),
        "full_name": "SimpleQLRouting"
    },
    "GEOQL": {
        "hatch": "",
        "markers": "s",
        "linestyle": "-",
        "color": 'tab:red',
        "label": "GEOQL",
        "x_ticks_positions": np.array(np.arange(5, 35, 5)),
        "full_name": "GeoQLRouting"
    },
    "QLS": {
        "hatch": "",
        "markers": "s",
        "linestyle": "-",
        "color": 'tab:purple',
        "label": "QLS",
        "x_ticks_positions": np.array(np.arange(5, 35, 5)),
        "full_name": "QLStepwiseRoutingProtocol"
    }
}