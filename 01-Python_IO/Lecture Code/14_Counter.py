from collections import Counter

l1 = [1,2,3,4,1,2,6,7,3,8,1]
print(Counter(l1))
l2 = ['a', 'b', 'c', 'd', 'a' , 'c', 'c']
print(Counter(l2))
cnt = Counter(l1)
print(cnt[1])

print(cnt.most_common())
cnt = Counter({1:3,2:4})
deduct = {1:1, 2:2}
cnt.subtract(deduct)
print(cnt)
