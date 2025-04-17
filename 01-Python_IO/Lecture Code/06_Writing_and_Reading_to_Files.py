with open("sample_text1.txt",'w',encoding = 'utf-8') as f:
   f.write("Thi is my first file created. \n")
   f.write("This is the second line file\n\n")
   f.write("Last but not least\n")
   f.close()
# %% -------------------More on Read-------------------
f = open("sample_text1.txt",'r',encoding = 'utf-8')
f.read(4)    # read the first 4 data
f.read(4)    # read the next 4 data
f.read()     # read in the rest till end of file
f.read()     # further reading returns empty sting
f.tell()     # get the current file position
f.seek(0)    # bring file cursor to initial position
print(f.read())
f.close()
