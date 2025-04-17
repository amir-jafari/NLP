import itertools

S1 = ['A', 'B', 'C', 'D']
selections = [True, False, True, False]
result = itertools.compress(S1, selections)
for each in result:
    print(each)

S2 = itertools.islice(S1, 2)
for each in S2:
    print(each)

S3 = itertools.permutations(S1)
for each in S3:
    print(each)