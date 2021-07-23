import itertools
for i in itertools.repeat("spam", 10):
    print(i)

S1 = ['A', 'B', 'C', 'D', 'E',]
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
for each in itertools.zip_longest(S1, data, fillvalue=None):
    print(each)

result = itertools.combinations_with_replacement(S1, 2)
for each in result:
    print(each)