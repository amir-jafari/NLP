import os
os.mkdir('test_dir')

orig_dir = os.getcwd()
print(orig_dir)
os.chdir(orig_dir + "\\\\" + 'test_dir')
new_dir = os.getcwd()
print(new_dir)
os.chdir(orig_dir)

os.rename('test_dir', 'new_test_dir')

l1 = os.listdir('new_test_dir')
print(l1)
l2 = os.listdir(os.curdir)
print(l2)
print(l2.sort())
