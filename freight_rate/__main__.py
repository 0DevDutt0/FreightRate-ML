"""python -m freight_rate {eda,validate,predict,score,report,all}"""

from __future__ import annotations

import argparse
import logging
import time

from . import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m freight_rate", description=__doc__)
    parser.add_argument("command", choices=["eda", "validate", "predict", "score", "report", "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    started = time.time()
    steps = ["eda", "validate", "predict", "score", "report"] if args.command == "all" else [args.command]
    ws = pipeline.load_workspace() if set(steps) & {"eda", "validate", "predict"} else None
    for step in steps:
        if step == "eda":
            pipeline.run_eda(ws)
        elif step == "validate":
            pipeline.run_validation(ws)
        elif step == "predict":
            pipeline.run_predict(ws)
        elif step == "score":
            pipeline.run_scorer()
        elif step == "report":
            from .report import build_report
            build_report()
    logging.getLogger("freight_rate").info("done in %.0fs", time.time() - started)


if __name__ == "__main__":
    main()
