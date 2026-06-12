"""Train generative models from YAML configuration files."""

import argparse

from dl_base import count_parameters, get_device
from project3_catgen.config import load_config
from project3_catgen.experiments.utils import setup_dcgan_experiment, setup_vae_experiment


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each seed from its last checkpoint",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    config = load_config(args.config)

    for seed in args.seeds:
        print(f"Running {config.run_name} | seed={seed} | device={get_device().type}")

        if config.model_name == "dcgan":
            trainer, train_loader = setup_dcgan_experiment(config, seed)
            print(
                "  Parameters: "
                f"G={count_parameters(trainer.generator):,}, "
                f"D={count_parameters(trainer.discriminator):,}"
            )
        elif config.model_name == "vae":
            trainer, train_loader = setup_vae_experiment(config, seed)
            print(f"  Parameters: {count_parameters(trainer.vae):,}")
        else:
            raise ValueError(f"Unknown model_name: {config.model_name!r}")

        if args.resume:
            trainer.load_checkpoint("last")

        trainer.fit(
            train_loader,
            config.training.num_epochs,
            config.project_name,
            config.run_name,
            f"{config.run_name}_seed{seed}",
            sample_every=config.training.sample_every,
            checkpoint_every=config.training.checkpoint_every,
            wandb_mode=args.wandb_mode,
            wandb_config=config.model_dump(mode="json"),
        )


if __name__ == "__main__":
    main()
