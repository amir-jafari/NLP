import itertools
import operator
data = [1, 2, 3, 4, 5]
states = ['Newyork', 'Virginia', 'DC', 'Texas']
[print(each) for each in states]
result = itertools.accumulate(data, operator.mul)
for each in result:
    print(each)
print(operator.mul(1,9))
print(operator.pow(2,4))
print(help(operator))

result = itertools.combinations(states, 2)
for each in result:
    print(each)
for i in itertools.count(10,3):
    print(i)
    if i > 20:
        break


