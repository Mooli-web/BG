"""Command-line entry points for training, evaluation, and the GUI."""

from __future__ import annotations

import argparse

from .training import TrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bg",
        description="Train a self-play RL backgammon agent and play against it.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train with self-play PPO")
    train_parser.add_argument(
        "--total-steps",
        type=int,
        default=5_000_000,
        help="total checker-move decisions; default: 5,000,000",
    )
    train_parser.add_argument("--num-envs", type=int, default=16)
    train_parser.add_argument("--rollout-steps", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--gamma", type=float, default=0.995)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--update-epochs", type=int, default=4)
    train_parser.add_argument("--minibatch-size", type=int, default=1024)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument("--hidden-size", type=int, default=256)
    train_parser.add_argument("--residual-blocks", type=int, default=3)
    train_parser.add_argument(
        "--reward-shaping",
        type=float,
        default=0.0,
        help="optional pip-progress reward; 0 keeps a sparse +1/-1 game reward",
    )
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--checkpoint-dir", default="checkpoints")
    train_parser.add_argument("--save-interval", type=int, default=50, help="save every N updates")
    train_parser.add_argument("--log-interval", type=int, default=10, help="print every N updates")
    train_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train_parser.add_argument(
        "--resume",
        default=None,
        help="resume from a checkpoint; --total-steps is the new total target",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="measure a checkpoint against a random player")
    from .evaluate import add_evaluate_arguments

    add_evaluate_arguments(evaluate_parser)

    play_parser = subparsers.add_parser("play", help="open the pygame human-vs-AI board")
    play_parser.add_argument("--checkpoint", required=True, help="path to checkpoints/latest.pt")
    play_parser.add_argument("--human", choices=["white", "black"], default="white")
    play_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    play_parser.add_argument("--seed", type=int, default=123)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainConfig(
            total_steps=args.total_steps,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            hidden_size=args.hidden_size,
            residual_blocks=args.residual_blocks,
            reward_shaping=args.reward_shaping,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            device=args.device,
            resume=args.resume,
        )
        train(config)
        return 0

    if args.command == "evaluate":
        from .evaluate import evaluation_from_args

        return evaluation_from_args(args)

    if args.command == "play":
        from .ui import run_game

        human_player = 0 if args.human == "white" else 1
        run_game(
            checkpoint_path=args.checkpoint,
            human_player=human_player,
            device_name=args.device,
            seed=args.seed,
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
