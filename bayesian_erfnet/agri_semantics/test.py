from os.path import join, dirname, abspath
import os
import click
import yaml
from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import torch
import matplotlib.pyplot as plt
import numpy as np


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config", "config.yaml"),
)
@click.option("--checkpoint", "-ckpt", type=str, help="path to checkpoint file (.ckpt)", required=True)
@click.option("--output-dir", "-o", type=str, help="directory to save output images", default="output_images")
def main(config, checkpoint, output_dir):
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    # Load data and model
    data = get_data_module(cfg)
    model = get_model(cfg)

    # Load checkpoint
    checkpoint_data = torch.load(checkpoint)
    if 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data

    # Adjust state_dict keys if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v  # Remove 'model.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    tb_logger = pl_loggers.TensorBoardLogger("experiments/" + cfg["experiment"]["id"], default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(logger=tb_logger, accelerator="gpu", devices=cfg["train"]["n_gpus"])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Test and visualize results
    results = trainer.test(model, data)


if __name__ == "__main__":
    main()
