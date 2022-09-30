from collections import Counter
import collections
from typing import DefaultDict
import pandas as pd
# --------------------------------Q1-------------------------------------------------------------------------------------



# --------------------------------Q2-------------------------------------------------------------------------------------





# --------------------------------Q3-------------------------------------------------------------------------------------





# --------------------------------Q4-------------------------------------------------------------------------------------





# --------------------------------Q5-------------------------------------------------------------------------------------



def main() -> None:
    sentence = []
    with open("sample.txt", "r") as file:
        data = file.read().split(sep= ".")
                                                                            # for line in file:
                                                                                #     line_strip:str = line.strip()
                                                                                #     if len(line_strip) > 15 and line_strip[-1] == ".":
                                                                                #         print(line_strip)
                                                                                #         d:DefaultDict = collections.defaultdict(int)
                                                                                #         for e in line_strip:
                                                                                #             if not e.isalpha():
                                                                                #                 d[e] += 1
                                                                                #         if d.get(" "):
                                                                                #             d.pop(" ")
                                                                                #         print(d)  
                                                                                #         sentence.append(line_strip)
                                                                                #         dict = {line_strip: len(line_strip)}
    df = pd.DataFrame(data= sentence, columns=["sentence", "len"])            



if __name__ == "__main__":
    main()