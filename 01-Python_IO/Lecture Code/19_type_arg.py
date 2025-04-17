import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, required=True, help="define the number")
args = parser.parse_args()
n = args.n
for i in range(n):
    print(random.randint(-100, 100))