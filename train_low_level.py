from __future__ import annotations

from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env
from fr5_rh56e2_dgrasp_rl.train_loop import build_arg_parser, train_main


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    run_dir = train_main(args)
    print(f"Training artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
