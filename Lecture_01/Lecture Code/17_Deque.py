from collections import deque

list = ["a","b","c"]
deq = deque(list)
print(deq)

deq.append("d")
deq.appendleft("e")
print(deq)

deq.pop()
deq.popleft()
print(deq)
print(deq.clear())

list = ["a","b","c"]
deq = deque(list)
print(deq.count("a"))