import shutil
import os
import subprocess
shutil.rmtree('new_test_dir')

os.mkdir('test')
shutil.copytree('test', 'test1')

os.system('python 01_Strings.py')
subprocess.call('python 01_Strings.py', shell=True)

path = os.path.join(os.getcwd(), 'test')
print(path)
foldername, basename = os.path.split(path)
print(foldername)
print(basename)

shutil.rmtree('test')
shutil.rmtree('test1')


