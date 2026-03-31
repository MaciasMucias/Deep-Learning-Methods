import argparse

from dl_base import get_device
from project1_cinic10.experiments.utils import setup_experiment
from project1_cinic10.config import load_config


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    parser.add_argument("--seeds", type=int, nargs="+", help="seeds to use")
    return parser.parse_args()



def main() -> None:
    args = get_args()
    config = load_config(args.config)

    for seed in args.seeds:
        print(f"Running {config.run_name} with seed {seed} on {get_device().type}")
        trainer, train_loader, val_loader, _ = setup_experiment(config, seed)
        trainer.fit(train_loader, val_loader, config.training.num_epochs, config.project_name, f"{config.run_name}_seed{seed}")

if __name__ == "__main__":
    main()