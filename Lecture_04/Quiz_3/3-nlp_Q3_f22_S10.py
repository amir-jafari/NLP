# ------------------Import Library----------------------------
import sys
import pandas as pd
import spacy
# --------
# ----------Main Loop----------------------------


# ------------------Part i----------------------------

# ------------------Part ii----------------------------

# ------------------Part iii----------------------------

def main() -> int:
    nlp = spacy.load("en_core_web_sm") 
    df:pd.DataFrame = pd.read_excel("NER.xlsx")
    text = [text for text in df["Sentences"]]
    text_str:str = " ".join(text)
    doc = nlp(text_str)
    loc = []
    for ent in doc.ents:
        if (ent.label == "GPE"):
            loc.append(ent.text)
    print(loc)
    return 0

if __name__ == "__main__":
    sys.exit(main())