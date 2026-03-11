from __future__ import annotations

from fr5_rh56e2_dgrasp_rl.evaluate import build_arg_parser, evaluate_main
from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    eval_dir = evaluate_main(args)
    print(f"Evaluation artifacts written to: {eval_dir}")


if __name__ == "__main__":
    main()
