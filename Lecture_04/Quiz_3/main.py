import sys
import spacy

def main() -> int:
    nlp = spacy.load("en_core_web_sm") 
    
    return 0

if __name__ == "__main__":
    sys.exit(main())