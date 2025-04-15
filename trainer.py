import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    log_dir = args["log_dir"]
    logs_name = "{}/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'], args['log_name'])
    logs_name = os.path.join(log_dir, logs_name)

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = os.path.join(log_dir, "{}/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args['log_name'],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    ))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    # This will store the results for different temperatures
    temperature_results = {}

    # Test different temperatures
    temperatures = [0.07, 0.1, 0.5]
    for temperature in temperatures:
        # Set temperature in the model
        model.set_temperature(temperature)

        cnn_curve = {"top1": [], "top5": []}
        metrics = {
            "final_accuracy": None,
            "average_forgetting": None,
            "training_time": None,
        }

        start_time = time.time()

        for task in range(data_manager.nb_tasks):
            logging.info("All params: {}".format(count_parameters(model._network)))
            logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
            model.incremental_train(data_manager)
            cnn_accy, nme_accy = model.eval_task()

            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

            cnn_accy, nme_accy = model.eval_task(only_new=True)
            cnn_accy, nme_accy = model.eval_task(only_old=True)

            model.after_task()
            if args["is_task0"]:
                break

        end_time = time.time()
        metrics["training_time"] = end_time - start_time
        metrics["final_accuracy"] = cnn_curve["top1"][-1]
        metrics["average_forgetting"] = sum(
            [
                max(cnn_curve["top1"][:i]) - acc
                for i, acc in enumerate(cnn_curve["top1"], start=1)
            ]
        ) / len(cnn_curve["top1"])

        logging.info("Metrics for temperature {}: {}".format(temperature, metrics))

        # Store the results for plotting
        temperature_results[temperature] = metrics["average_forgetting"]

    # Plot the temperature vs average forgetting graph
    plot_temperature_vs_forgetting(temperature_results)


def plot_temperature_vs_forgetting(temperature_results):
    # Convert the dictionary to lists for plotting
    temperatures = list(temperature_results.keys())
    average_forgetting = list(temperature_results.values())

    plt.figure(figsize=(6, 4))
    plt.plot(temperatures, average_forgetting, marker='o', linestyle='-', color='royalblue')
    plt.xlabel('Temperature (τ)', fontsize=12)
    plt.ylabel('Average Forgetting (%)', fontsize=12)
    plt.title('Effect of Temperature on Average Forgetting in FCS', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device_id in device_type:
        if device_id == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device_id))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
