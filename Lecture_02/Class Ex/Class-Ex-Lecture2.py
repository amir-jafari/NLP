# =================================================================
# Class_Ex1:
# Use NLTK Book fnd which the related Sense and Sensibility.
# Produce a dispersion plot of the four main protagonists in Sense and Sensibility:
# Elinor, Marianne, Edward, and Willoughby. What can you observe about the different
# roles played by the males and females in this novel? Can you identify the couples?
# Explain the result of plot in a couple of sentences.
# ----------------------------------------------------------------
import sys

def Ex1() -> None:
    from nltk.book import text2

    def main() -> None:
        text2.dispersion_plot(
            ["Elinor", "Marianne", "Edward", "Willoughby"]
        )
        print("\nApprantly, females play a huge part in this novel as females "
            "have significantly more words")
    
    main()


# =================================================================
# Class_Ex2:
# What is the difference between the following two lines of code? Explain in details why?
# Make up and example base don your explanation.
# Which one will give a larger value? Will this be the case for other texts?
# 1- sorted(set(w.lower() for w in text1))
# 2- sorted(w.lower() for w in set(text1))
# ----------------------------------------------------------------

def Ex2() -> None:
    from nltk.book import text1

    def main() -> None:
        print("\nSecond will get a larger value because")
    
    main()



# =================================================================
# Class_Ex3:
# Find all the four-letter words in the Chat Corpus (text5).
# With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
#
# ----------------------------------------------------------------

def Ex3() -> None:

    from nltk.book import text5
    from typing import Set
    from nltk import FreqDist

    def main() -> None:
        Value_set:Set = set(text5)  
        four_words = [words for words in Value_set if len(words) == 4]
        freq_dist = FreqDist(four_words)
        print(freq_dist.pprint())
    main()



# =================================================================
# Class_Ex4:
# Write expressions for finding all words in text6 that meet the conditions listed below.
# The result should be in the form of a list of words: ['word1', 'word2', ...].
# a. Ending in ise  
# b. Containing the letter z
# c. Containing the sequence of letters pt
# d. Having all lowercase letters except for an initial capital (i.e., titlecase)
# ----------------------------------------------------------------


def Ex4() -> None:
    from nltk.book import text6
    from typing import Set

    def main() -> None:
        end_set:Set = set(text6)  
        end = [words for words in end_set if words[-3:] == "ise"]
        print(f"Words end in 'ise' are {end}")

        text6.
        main()



# =================================================================
# Class_Ex5:
#  Read in the texts of the State of the Union addresses, using the state_union corpus reader.
#  Count occurrences of men, women, and people in each document.
#  What has happened to the usage of these words over time?
# Since there would be a lot of document use every couple of years.
# ----------------------------------------------------------------

def Ex5() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()




# =================================================================
# Class_Ex6:
# The CMU Pronouncing Dictionary contains multiple pronunciations for certain words.
# How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?
#
#
# ----------------------------------------------------------------

def Ex6() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()



# =================================================================
# Class_Ex7:
# What percentage of noun synsets have no hyponyms?
# You can get all noun synsets using wn.all_synsets('n')
#
# ----------------------------------------------------------------

def Ex7() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()



# =================================================================
# Class_Ex8:
# Write a program to find all words that occur at least three times in the Brown Corpus.
# USe at least 2 different method.
# ----------------------------------------------------------------


def Ex8() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()



# =================================================================
# Class_Ex9:
# Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
# Test it on Brown corpus (humor), Gutenberg (whitman-leaves.txt).
# Did you find any strange word in the list? If yes investigate the cause?
# ----------------------------------------------------------------


def Ex9() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()


# =================================================================
# Class_Ex10:
# Write a program to create a table of word frequencies by genre, like the one given in 1 for modals.
# Choose your own words and try to find words whose presence (or absence) is typical of a genre. Discuss your findings.

# ----------------------------------------------------------------


def Ex10() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()


# =================================================================
# Class_Ex11:
#  Write a utility function that takes a URL as its argument, and returns the contents of the URL,
#  with all HTML markup removed. Use from urllib import request and
#  then request.urlopen('http://nltk.org/').read().decode('utf8') to access the contents of the URL.
# ----------------------------------------------------------------


def Ex11() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()




# =================================================================
# Class_Ex12:
# Read in some text from a corpus, tokenize it, and print the list of all
# wh-word types that occur. (wh-words in English are used in questions,
# relative clauses and exclamations: who, which, what, and so on.)
# Print them in order. Are any words duplicated in this list,
# because of the presence of case distinctions or punctuation?
# Note Use: Gutenberg('bryant-stories.txt')
# ----------------------------------------------------------------


def Ex12() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()

# =================================================================
# Class_Ex13:
# Write code to access a  webpage and extract some text from it.
# For example, access a weather site and extract  a feels like temprature..
# Note use the following site https://darksky.net/forecast/40.7127,-74.0059/us12/en
# ----------------------------------------------------------------


def Ex13() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()


# =================================================================
# Class_Ex14:
# Use the brown tagged sentes corpus news.
# make a test and train sentences and then  use bigram tagger to train it.
# Then evlaute the trained model.
# ----------------------------------------------------------------

def Ex14() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()





# =================================================================
# Class_Ex15:
# Use sorted() and set() to get a sorted list of tags used in the Brown corpus, removing duplicates.
# ----------------------------------------------------------------

def Ex15() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()



# =================================================================
# Class_Ex16:
# Write programs to process the Brown Corpus and find answers to the following questions:
# 1- Which nouns are more common in their plural form, rather than their singular form? (Only consider regular plurals, formed with the -s suffix.)
# ----------------------------------------------------------------

def Ex16() -> None:

    def solve() -> None:
        pass

    def main() -> None:
        pass
    
    main()



# =================================================================

def main() -> int:

    # print(20*'-' + 'Begin Q1' + 20*'-')
    # Ex1()
    # print(20*'-' + 'End Q1' + 20*'-')

    # print(20*'-' + 'Begin Q2' + 20*'-')
    # Ex2()
    # print(20*'-' + 'End Q2' + 20*'-')

    # print(20*'-' + 'Begin Q3' + 20*'-')
    # Ex3()
    # print(20*'-' + 'End Q3' + 20*'-')

    print(20*'-' + 'Begin Q4' + 20*'-')
    Ex4()
    print(20*'-' + 'End Q4' + 20*'-')

    # print(20*'-' + 'Begin Q5' + 20*'-')
    # Ex5()
    # print(20*'-' + 'End Q5' + 20*'-')

    # print(20*'-' + 'Begin Q6' + 20*'-')
    # Ex6()
    # print(20*'-' + 'End Q6' + 20*'-')

    # print(20*'-' + 'Begin Q7' + 20*'-')
    # Ex7()
    # print(20*'-' + 'End Q7' + 20*'-')

    # print(20*'-' + 'Begin Q8' + 20*'-')
    # Ex8()
    # print(20*'-' + 'End Q8' + 20*'-')

    # print(20*'-' + 'Begin Q9' + 20*'-')
    # Ex9()
    # print(20*'-' + 'End Q9' + 20*'-')

    # print(20*'-' + 'Begin Q10' + 20*'-')
    # Ex10()
    # print(20*'-' + 'End Q10' + 20*'-')

    # print(20*'-' + 'Begin Q11' + 20*'-')
    # Ex11()
    # print(20*'-' + 'End Q11' + 20*'-')

    # print(20*'-' + 'Begin Q12' + 20*'-')
    # Ex12()
    # print(20*'-' + 'End Q12' + 20*'-')

    # print(20*'-' + 'Begin Q13' + 20*'-')
    # Ex13()
    # print(20*'-' + 'End Q13' + 20*'-')

    # print(20*'-' + 'Begin Q14' + 20*'-')
    # Ex14()
    # print(20*'-' + 'End Q14' + 20*'-')

    # print(20*'-' + 'Begin Q15' + 20*'-')

    # print(20*'-' + 'End Q15' + 20*'-')

    # print(20*'-' + 'Begin Q16' + 20*'-')

    # print(20*'-' + 'End Q16' + 20*'-')

    return 0

if __name__ == "__main__":
    sys.exit(main())
