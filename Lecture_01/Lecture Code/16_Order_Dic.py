from collections import OrderedDict
from collections import Counter

od = OrderedDict()
od['c'] = 1
od['b'] = 2
od['a'] = 3
print(od)
for key, value in od.items():
    print(key, value)


list = ["a","c","c","a","b","a","a","b","c"]
cnt = Counter(list)
od = OrderedDict(cnt.most_common())
for key, value in od.items():
    print(key, value)