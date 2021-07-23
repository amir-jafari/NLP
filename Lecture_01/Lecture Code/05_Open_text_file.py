f = open("sample_text.txt","w+")
for i in range (20):
    f.write('This is a line {} \n'.format(i+1))
f.close()
f = open("sample_text.txt", 'a+')
for i in range (2):
    f.write('Append a line {} \n'.format(i+1))
f.close()