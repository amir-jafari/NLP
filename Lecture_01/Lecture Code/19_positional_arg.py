import argparse
parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('age')
args = parser.parse_args()
print('{}'.format(args.name) + ' is '  +'{}'.format(args.age) + ' years old.')