f = open("sample_text1.txt",'r',encoding = 'utf-8')
for line in f:
   print(line)
f.seek(0)
print(f.readline())
print(f.readline())
print(f.readline())
print(f.readline())
f.seek(0)
print(f.readlines())
f.close()


