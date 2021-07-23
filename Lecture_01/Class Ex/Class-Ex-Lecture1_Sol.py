# =================================================================
# Class_Ex1: 
# Write a function that prints all the chars from string1 that appears in string2.
# Note: Just use the Strings functionality no packages should be used.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q1' + 20*'-')
def in_both(word1, word2):
    for letter in word1:
        if letter in word2:
            print (letter)

in_both('apples', 'oranges')
print(20*'-' + 'End Q1' + 20*'-')

# =================================================================
# Class_Ex2: 
# Write a function that counts the numbers of a particular letter in a string.
# For example count the number of letter a in abstract.
# Note: Compare your function with a count method
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q2' + 20*'-')
def COUNT(Word, let):
    word = Word
    count = 0
    for letter in word:
        if letter == let:
            count = count + 1
    return print (count)

COUNT('Bannana', 'a')
print('Bannana'.count('a'))
print(20*'-' + 'End Q2' + 20*'-')
# =================================================================
# Class_Ex3: 
# Write a function that reads the Story text and finds the strings in the curly brackets.
# Note: You are allowed to use the strings methods
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q3' + 20*'-')
def getKeys(formatString):
    '''
    formatString is a format string with embedded dictionary keys.
    Return a list containing all the keys from the format string.
    '''
    keyList = list()
    end = 0
    repetitions = formatString.count('{')
    for i in range(repetitions):
        start = formatString.find('{', end) + 1
        end = formatString.find('}', start)
        key = formatString[start : end]
        keyList.append(key)
    return keyList
Story = """
Once upon a time, deep in an ancient jungle,there lived a {animal}. This {animal} liked to eat {food}, but the jungle had 
very little {food} to offer. One day, an explorer found the {animal} and discovered it liked {food}. The explorer took the
{animal} back to {city}, where it could eat as much {food} as it wanted. However, the {animal} became homesick, so the
explorer brought it back to the jungle, leaving a large supply of {food}.

The End
"""
print(getKeys(Story))
print(20*'-' + 'End Q3' + 20*'-')
# =================================================================
# Class_Ex4: 
# Write a function that read the first n lines of a file.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q4' + 20*'-')
def file_read_from_head(fname, nlines):
    from itertools import islice
    with open(fname) as f:
        for line in islice(f, nlines):
            print(line)


file_read_from_head('test_1.txt', 2)
print(20*'-' + 'End Q4' + 20*'-')
# =================================================================
# Class_Ex5:
# Write a function that read a file line by line and store it into a list.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q5' + 20*'-')
def file_read(fname):
    with open(fname) as f:
        # Content_list is the list that contains the read lines.
        content_list = f.readlines()
        print(content_list)


file_read('test_1.txt')
print(20*'-' + 'End Q5' + 20*'-')

# =================================================================
# Class_Ex6:
# Write a function that read two text files and combine each line from first
# text file with the corresponding line in second text file.
# Use T1.txt and T2.txt as sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q6' + 20*'-')
with open('T1.txt') as fh1, open('T2.txt') as fh2:
    for line1, line2 in zip(fh1, fh2):
        print(line1+line2)
print(20*'-' + 'End Q6' + 20*'-')
# =================================================================
# Class_Ex7:
# Write a function that create a text file where all letters of English alphabet
# put together by number of letters on each line (use n as argument in the function).
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q7' + 20*'-')
import string
def letters_file_line(n):
   with open("test_2.txt", "w") as f:
       alphabet = string.ascii_uppercase
       letters = [alphabet[i:i + n] + "\n" for i in range(0, len(alphabet), n)]
       f.writelines(letters)
letters_file_line(3)
print(' It is created. Done!')

print(20*'-' + 'End Q7' + 20*'-')
# =================================================================
# Class_Ex8:
# Write a function that reads a text file and count number of words.
# Note: USe test_1.txt as a sample text.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q8' + 20*'-')
def count_words(filepath):
   with open(filepath) as f:
       data = f.read()
       data.replace(",", " ")
       return len(data.split(" "))
print(count_words("test_1.txt"))
print(20*'-' + 'End Q8' + 20*'-')
# =================================================================
# Class_Ex9:
# Write a script that go over over elements and repeat it each as many times as its count.
# Sample Output = ['o' ,'o', 'o', 'g' ,'g', 'f']
# Use Collections
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q9' + 20*'-')

from collections import Counter
c = Counter(p=4, q=2, r=0, s=-2)
print(list(c.elements()))

print(20*'-' + 'End Q9' + 20*'-')
# =================================================================
# Class_Ex10:
# Write a program that appends couple of integers to a list
# and then with certain index start the the list over that index.
# Note: use deque
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q10' + 20*'-')

import collections
dq_object = collections.deque()
dq_object.append(2)
dq_object.append(4)
dq_object.append(6)
dq_object.append(8)
dq_object.append(10)
print("Deque before rotation:")
print(dq_object)
dq_object.rotate()
print("\nDeque after 1 positive rotation:")
print(dq_object)
dq_object.rotate(2)
print("\nDeque after 2 positive rotations:")
print(dq_object)


print(20*'-' + 'End Q10' + 20*'-')
# =================================================================
# Class_Ex11:
# Write a script using os command that finds only directories, files and all directories, files in a  path.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q11' + 20*'-')
import os
print("Only directories:")
path = os.getcwd()
print([ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ])
print("\nOnly files:")
print([ name for name in os.listdir(path) if not os.path.isdir(os.path.join(path, name)) ])
print("\nAll directories and files :")
print([ name for name in os.listdir(path)])

print(20*'-' + 'End Q11' + 20*'-')
# =================================================================
# Class_Ex12:
# Write a script that create a file and write a specific text in it and rename the file name.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q12' + 20*'-')

import os
with open('a.txt', 'w') as f:
   f.write('Python program to create a symbolic link and read it to decide the original file pointed by the link.')
print('\nInitial file/dir name:', os.listdir())
with open('a.txt', 'r') as f:
   print('\nContents of a.txt:', repr(f.read()))
os.rename('a.txt', 'b.txt')
print('\nAfter renaming initial file/dir name:', os.listdir())
with open('b.txt', 'r') as f:
   print('\nContents of b.txt:', repr(f.read()))

print(20*'-' + 'End Q12' + 20*'-')
# =================================================================
# Class_Ex13:
#  Write a script  that scan a specified directory find which is  file and which is a directory.
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q13' + 20*'-')
import os
root = os.getcwd()
for entry in os.scandir(root):
   if entry.is_dir():
       typ = 'dir'
   elif entry.is_file():
       typ = 'file'
   elif entry.is_symlink():
       typ = 'link'
   else:
       typ = 'unknown'
   print('{name} {typ}'.format(
       name=entry.name,
       typ=typ,
   ))







print(20*'-' + 'End Q13' + 20*'-')
# =================================================================
# Class_Ex14:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q14' + 20*'-')








print(20*'-' + 'End Q14' + 20*'-')

# =================================================================
# Class_Ex15:
#
# ----------------------------------------------------------------
print(20*'-' + 'Begin Q15' + 20*'-')








print(20*'-' + 'End Q15' + 20*'-')