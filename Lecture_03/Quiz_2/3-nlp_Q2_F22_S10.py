# ------------------Import Library----------------------------





#-------------------Q1------------------------------





#-------------------Q2------------------------------




#-------------------Q3------------------------------




#-------------------Q4------------------------------




#-------------------Q5------------------------------

import sys
import nltk
from nltk.corpus import genesis
from nltk.text import Text



def main() -> int:
    fileids = genesis.fileids()
    
    with open("text.txt", "w+") as file:
        file.write(genesis.raw())
        
    with open("text.txt", "r") as file:
        text = ""
        for line in file:
            text += line
        words = nltk.word_tokenize(text)
        sents = nltk.sent_tokenize(text)

        print(f"length of sentence is {len(sents)}\n")
        sentences = text.split('. ')
        words_in_sentences = [sentence.split(' ') for sentence in sentences]
        print(f"words in a sentence is {words_in_sentences}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
