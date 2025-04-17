import shutil
import os
import subprocess
shutil.rmtree('new_test_dir')

os.makedirs('test', exist_ok=True)
shutil.copytree('test', 'test1')

os.system('python3 01_Strings.py')
subprocess.call('python3 01_Strings.py', shell=True)

path = os.path.join(os.getcwd(), 'test')
print(path)
foldername, basename = os.path.split(path)
print(foldername)
print(basename)

shutil.rmtree('test')
shutil.rmtree('test1')


