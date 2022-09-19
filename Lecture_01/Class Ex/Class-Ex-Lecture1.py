# =================================================================
# Class_Ex1:
# Write a function that prints all the chars from string1 that appears in string2.
# Note: Just use the Strings functionality no packages should be used.
# ----------------------------------------------------------------
from typing import Generator, List, Set, Deque




def Ex_1() -> None:

    def solve(str_1:str, str_2:str) -> None:
        str_1:Set[str] = set(e for e in str_1 if e.isalnum())
        str_2:Set[str] = set(e for e in str_2 if e.isalnum())
        str_3:Set[str] = str_1.intersection(str_2)
        if len(str_3) == 0:
            print("\nThere are no common strings")
        else:
            print(str_3)

    def main() -> None:
        str_1:str = input("Please input the first string: ")
        str_2:str = input("Please input the second string: ")
        solve(str_1, str_2)

    main()

# =================================================================
# Class_Ex2:
# Write a function that counts the numbers of a particular letter in a string.
# For example count the number of letter a in abstract.
# Note: Compare your function with a count method
# ----------------------------------------------------------------

def Ex_2() -> None:

    def solve(str_1:str) -> None:
        str_1_alnum:List[str] = [e for e in str_1 if e.isalnum()]
        char:str = input("what is the character you want to calculate: ")
        if char in str_1_alnum:
            print(f"\nCharacter '{char}' has appeared {str_1_alnum.count(char)} times in '{str_1}'.")
        else:
            char = input(f"{char} is not in the string please reenter a character: ")
            print(f"\nCharacter '{char}' has appeared {str_1_alnum.count(char)} times in '{str_1}'.")

    def main() -> None:
        str_1:str = input("What is the string: ")
        solve(str_1)
        
    main()

# =================================================================
# Class_Ex3:
# Write a function that reads the Story text and finds the strings in the curly brackets.
# Note: You are allowed to use the strings methods
# Copy a text from wiki and add some curly bracket in the text call the string Story.
# ----------------------------------------------------------------
                                                # what does this question mean
def Ex_3() -> None:

    def solve() -> None:
        
        pass

    def main() -> None:

        pass

    main()

# =================================================================
# Class_Ex4:
# Write a function that read the first n lines of a file.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------

def Ex_4() -> None:

    def solve(num:int) -> None:
        with open("test_1.txt", "r") as file:
            lines:int = len(file.readlines())
            file.seek(0)
            if num > lines:
                num = int(input(f"\nThere are only {lines} lines in the file, please reenter: "))
            print(f"\nThe {num} lines are: ")
            for i in range(num):
                line = next(file).strip()
                print(line)

    def main() -> None:

        num:int = int(input("How many lines do you want to read in a file: "))
        print()
        solve(num)

    main()

# =================================================================
# Class_Ex5:
# Write a function that read a file line by line and store it into a list.
# Use test_1.txt as sample text.
# ----------------------------------------------------------------

def Ex_5() -> None:

    def solve() -> None:
        with open("test_1.txt", "r") as file:
            print([line.rstrip() for line in file])

    def main() -> None:
        solve()

    main()

# =================================================================
# Class_Ex6:
# Write a function that read two text files and combine each line from first
# text file with the corresponding line in second text file.
# Use T1.txt and T2.txt as sample text.
# ----------------------------------------------------------------

def Ex_6() -> None:

    def solve() -> None:
        with open("T1.txt", "r") as file_1:
            with open("T2.txt", "r") as file_2:
                for _i, (line_1, line_2) in enumerate(zip(file_1, file_2)):
                    print(line_1.rstrip() + line_2.rstrip())
    def main() -> None:
        solve()
    
    main()

# =================================================================
# Class_Ex7:
# Write a function that create a text file where all letters of English alphabet
# put together by number of letters on each line (use n as argument in the function).
# Sample output
# function(3)
# ABC
# DEF
# ...
# ...
# ...
# ----------------------------------------------------------------

def Ex_7() -> None:

    import string

    def solve(num:int, alphabet:str) -> None:
        with open("Exercise_7.txt", "w") as file:
            for i in range(26//num):
                file.write(alphabet[i * num: (i+1) * num] + "\n")

    def main() -> None:
        num:int = int(input("How many letters do you want on each line: "))
        alphabet:str = string.ascii_uppercase
        solve(num, alphabet)

    main()


# =================================================================
# Class_Ex8:
# Write a function that reads a text file and count number of words.
# Note: USe test_1.txt as a sample text.
# ----------------------------------------------------------------

def Ex_8() -> None:

    def solve() -> None:
        with open("test_1.txt", "r") as file:
            words = file.read()
            print(f"There are {len(words.split())} words in the file.") # how to remove the punctuaion
    def main() -> None:
        solve()

    main()

# =================================================================
# Class_Ex9:
# Write a script that go over over elements and repeat it each as many times as its count.
# Sample Output = ['o' ,'o', 'o', 'g' ,'g', 'f']
# Use Collections
# ----------------------------------------------------------------

def Ex_9() -> None:                                         # what does it mean

    def solve() -> None:
        pass

    def main() -> None:
        solve()
    
    main()

# =================================================================
# Class_Ex10:
# Write a program that appends couple of integers to a list
# and then with certain index start the the list over that index.
# Note: use deque
# ----------------------------------------------------------------


def Ex_10() -> None:
    from collections import deque
    def solve(int_deq:Deque[int], index:int) -> None:
        i:int = 0
        while i< index:
            int_deq.popleft()
            i += 1
        print(int_deq)
    def main() -> None:
        int_deq:Deque[int] = deque(map(int, input("What are the intergers: ").split()))
        index:int = int(input("What is the index: "))
        solve(int_deq, index)
    main()

# =================================================================
# Class_Ex11:
# Write a script using os command that finds only directories, files and all directories, files in a  path.
# ----------------------------------------------------------------

def Ex_11() -> None:
    
    import os

    def files_gen(path) -> Generator:
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    def solve(path) -> None:
        dir_list:List[str] = []
        for all in os.listdir(path):
            if os.path.isdir(all):
                dir_list.append(all)
        print(f"These are all the directories in the path: {dir_list}")
        file_list:List[str] = []
        for file in files_gen(path):
            file_list.append(file)
        print(f"These are all the files in the path: {file_list}")
        print(f"These are all the directories and files in the path: {os.listdir(path)}")

    def main() -> None:
        path:str
        path = input("Please give a path to test (If you don't have one, simply enter. We will use the current working directory): ")
        if path == "":
            path = os.getcwd()
        solve(path)
        
    main()

# =================================================================
# Class_Ex12:
# Write a script that create a file and write a specific text in it and rename the file name.
# ----------------------------------------------------------------

def Ex_12() -> None:

    import os
    def solve(text:str, name:str) -> None:
        with open("Exercise_12.txt", "w") as file:
            file.write(text)
        os.rename("Exercise_12.txt", name)
    def main() -> None:
        text:str = input("What is the text you want to write into the file: ")
        name:str = input("What do you want to rename the filename to: ")
        if name.find(".txt"):
            name = name + ".txt"
        solve(text, name)

    main()



# =================================================================
# Class_Ex13:
#  Write a script  that scan a specified directory find which is  file and which is a directory.
# ----------------------------------------------------------------


                                                  # what's the difference between this and 11







# =================================================================
# main function

def main() -> None:
    print(20*'-' + 'Begin Q1' + 20*'-')
    Ex_1()
    print(20*'-' + 'End Q1' + 20*'-')

    print(20*'-' + 'Begin Q2' + 20*'-')
    Ex_2()
    print(20*'-' + 'End Q2' + 20*'-')

    # print(20*'-' + 'Begin Q3' + 20*'-')
    # Ex_3()
    # print(20*'-' + 'End Q3' + 20*'-')

    print(20*'-' + 'Begin Q4' + 20*'-')
    Ex_4()
    print(20*'-' + 'End Q4' + 20*'-')

    print(20*'-' + 'Begin Q5' + 20*'-') 
    Ex_5()
    print(20*'-' + 'End Q5' + 20*'-')

    print(20*'-' + 'Begin Q6' + 20*'-')
    Ex_6()
    print(20*'-' + 'End Q6' + 20*'-')

    print(20*'-' + 'Begin Q7' + 20*'-')
    Ex_7()
    print(20*'-' + 'End Q7' + 20*'-')

    print(20*'-' + 'Begin Q8' + 20*'-')
    Ex_8()
    print(20*'-' + 'End Q8' + 20*'-')

    # print(20*'-' + 'Begin Q9' + 20*'-')
    # Ex_9()
    # print(20*'-' + 'End Q9' + 20*'-')

    print(20*'-' + 'Begin Q10' + 20*'-')
    Ex_10()
    print(20*'-' + 'End Q10' + 20*'-')

    print(20*'-' + 'Begin Q11' + 20*'-')
    Ex_11()
    print(20*'-' + 'End Q11' + 20*'-')

    print(20*'-' + 'Begin Q12' + 20*'-')
    Ex_12()
    print(20*'-' + 'End Q12' + 20*'-')

    # print(20*'-' + 'Begin Q13' + 20*'-')    
    
    # print(20*'-' + 'End Q13' + 20*'-')

if __name__ == "__main__":
    main()