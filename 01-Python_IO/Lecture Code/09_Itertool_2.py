import itertools
for i in itertools.count(1,2):
    print(i)
    if i > 20:
        break

states = ['Newyork', 'Virginia', 'DC', 'Texas']
for index, city in enumerate(itertools.cycle(states)):
    print(city)
    if index==10:
        break

S1 = ['A', 'B', 'C', 'D', 'E']
S2 = ['F', 'G', 'H', 'I']
result = itertools.chain(S1, S2)
for each in result:
    print(each)

