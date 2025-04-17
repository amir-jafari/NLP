from collections import defaultdict

nums = defaultdict(int)
nums['one'] = 1
nums['two'] = 2
print(nums['three'])

count = defaultdict(int)
names = "John Julie Jack Ann Mike John John Jack Jack Jen Smith Jen Jen"
list = names.split(sep=' ')
for names in list:
    count[names] +=1
print(count)