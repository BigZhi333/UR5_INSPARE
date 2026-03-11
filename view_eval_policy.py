from __future__ import annotations

from fr5_rh56e2_dgrasp_rl.paths import configure_local_runtime_env
from fr5_rh56e2_dgrasp_rl.visualize_eval import build_arg_parser, visualize_eval_main


def main() -> None:
    configure_local_runtime_env()
    args = build_arg_parser().parse_args()
    visualize_eval_main(args)


if __name__ == "__main__":
    main()
