import argparse

from project1_cinic10.experiments.utils import setup_experiment
from project1_cinic10.config import load_config


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    parser.add_argument("--seed", type=int, help="specific seed to use")
    parser.add_argument("--checkpoint", type=str, choices=["best", "last"], help="checkpoint file selection")
    return parser.parse_args()



def main() -> None:
    args = get_args()
    config = load_config(args.config)

    trainer, _, _, test_loader = setup_experiment(config, args.seed)
    trainer.load_checkpoint(args.checkpoint)

    avg_loss, accuracy = trainer.test(test_loader, project_name=config.project_name, run_name=f"{config.run_name}_seed{args.seed}")
    print(f"loss: {avg_loss:.3f}, accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()