# ----------------------------------------------------------------------------------------------------------------------
import argparse, random, json, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--input"); ap.add_argument("--train"); ap.add_argument("--test")
ap.add_argument("--test-split", type=float, default=0.1)
args = ap.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
lines = pathlib.Path(args.input).read_text().splitlines()
random.shuffle(lines)
cut = int(len(lines) * (1 - args.test_split))

pathlib.Path(args.train).write_text("\n".join(
    json.dumps({"text": t}) for t in lines[:cut])
)
pathlib.Path(args.test).write_text("\n".join(
    json.dumps({"text": t}) for t in lines[cut:])
)
