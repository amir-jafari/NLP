# =================================================================
# Class_Hw1:
# Write a python script that reads a string from the user input and print the following
# i. Number of uppercase letters in the string.
# ii. Number of lowercase letters in the string
# iii. Number of digits in the string
# iv. Number of whitespace characters in the string
# ----------------------------------------------------------------

import collections
import itertools
from typing import DefaultDict, List


def Hw_1() -> None:

    def solve(string:str) -> None:
        print(f"\ni. Number of uppercase letters in the string: {sum(1 for char in string if char.isupper())}")
        print(f"\ni. Number of lowercase letters in the string: {sum(1 for char in string if char.islower())}")
        print(f"\ni. Number of digits in the string: {sum(1 for char in string if char.isnumeric())}")
        print(f"\ni. Number of white spaces in the string: {string.count(' ')}")

    def main() -> None:
        string:str = input("Please give a string: ")
        solve(string)

    main()

# =================================================================
# Class_Hw2:
# Write a python script that accepts a string then create a new string by shifting one position to
# left.
# Example: input : class 2021 output: lass 2021c
# ----------------------------------------------------------------

def Hw_2() -> None:

    def solve(string:str) -> None:
        print(f"The new string is {string[1:-1] + string[0]}")
    def main() -> None:
        string:str = input("What is the string: ")
        solve(string)

    main()

# =================================================================
# Class_Hw3 :
# Write a python script that a user input his name and program display its initials.
# Hint: Assuming, user always enter first name, middle name and last name.
# ----------------------------------------------------------------

def Hw_3() -> None:

    def solve(name:str) -> None:
        name_lst:List[str] = name.split()
        print(f"Your initials are: {name_lst[0][0] + name_lst[1][0] + name_lst[2][0]}")

    def main() -> None:
        name:str = input("What is your name: ")
        solve(name)

    main()    
# =================================================================
# Class_Hw4 :
# Write a python script that accepts a string to setup a passwords. The password must have the
# following requirements
# • The password must be at least eight characters long.
# • It must contain at least one uppercase letter.
# • It must contain at least one lowercase letter.
# • It must contain at least one numeric digit.
# ----------------------------------------------------------------

def Hw_4() -> None:

    def solve(string:str) -> None:
        while len(string) < 8 or not any(char.isupper() for char in string) or not any(char.islower() for char in string) or not any(char.isnumeric() for char in string):
            string = input(
                "\nplese reenter the password: \n"
                "• The password must be at least eight characters long.\n"
                "• It must contain at least one uppercase letter.\n"
                "• It must contain at least one lowercase letter.\n"
                "• It must contain at least one numeric digit.\n"
            )
        print("\nThe password has been set.")



    def main() -> None:
        string:str = input(
            "plese enter the password: \n"
            "• The password must be at least eight characters long.\n"
            "• It must contain at least one uppercase letter.\n"
            "• It must contain at least one lowercase letter.\n"
            "• It must contain at least one numeric digit.\n"
        )
        solve(string)

    main()

# =================================================================
# Class_Hw5 :
# Write a python script that reads a given string character by character and count the repeated
# characters then store it by length of those character(s).
# ----------------------------------------------------------------

def Hw_5() -> None:

    def solve(string:str) -> None:
        d:DefaultDict = collections.defaultdict(int)
        for c in string:
            d[c] += 1
        d.pop(" ")

        print(d)


    def main() -> None:
        string:str = input("What is the string: ")
        solve(string)
   
    main()

# =================================================================
# Class_Hw6 :
# Write a python script to find all lower and upper case combinations of a given string.
# Example: input: abc output: ’abc’, ’abC’, ’aBc’, ...
# ----------------------------------------------------------------

def Hw_6() -> None:

    def solve(string:str) -> None:
        list(map("".join, itertools.product(*zip(string.upper(), string.lower()))))

    def main() -> None:
        string:str = input("What is the string: ")
        solve(string)
    main()

# =================================================================
# Class_Hw7 :
# Write a python script that
# i. Read first n lines of a file.
# ii. Find the longest words.
# iii. Count the number of lines in a text file.
# iv. Count the frequency of words in a file.
# Hint: first create a test.txt file and dump some textual data in it. Then test your code.
# ----------------------------------------------------------------

# =================================================================
# main function

def main() -> None:
    # print(20*'-' + 'Begin Q1' + 20*'-')
    # Hw_1()
    # print(20*'-' + 'End Q1' + 20*'-')

    # print(20*'-' + 'Begin Q2' + 20*'-')
    # Hw_2()
    # print(20*'-' + 'End Q2' + 20*'-')

    # print(20*'-' + 'Begin Q3' + 20*'-')
    # Hw_3()
    # print(20*'-' + 'End Q3' + 20*'-')
    
    # print(20*'-' + 'Begin Q4' + 20*'-')
    # Hw_4()
    # print(20*'-' + 'End Q4' + 20*'-')
    
    # print(20*'-' + 'Begin Q5' + 20*'-')
    # Hw_5()
    # print(20*'-' + 'End Q5' + 20*'-')
    
    print(20*'-' + 'Begin Q6' + 20*'-')
    Hw_6()
    print(20*'-' + 'End Q6' + 20*'-')

if __name__ == "__main__":
    main()