from collections import Counter
import pandas as pd
# --------------------------------Q1-------------------------------------------------------------------------------------
f = open('sample.txt', 'r', encoding='utf-8')
data = f.read()
# --------------------------------Q2-------------------------------------------------------------------------------------
l = data.split(sep='.')
l_less_sent_10 = [x for x in l if len(x) > 15]
# --------------------------------Q3-------------------------------------------------------------------------------------
All_list_non_alphabet = [x for y in l for x in y if not x.isalpha()]
Cnt =Counter(All_list_non_alphabet)
print(Cnt.most_common(5))
# --------------------------------Q4-------------------------------------------------------------------------------------
count_comma = [x.count(Cnt.most_common(5)[1][0]) for x in l_less_sent_10 ]
count_quote = [x.count(Cnt.most_common(5)[2][0]) for x in l_less_sent_10 ]
count_prime = [x.count(Cnt.most_common(5)[3][0]) for x in l_less_sent_10 ]
count_dash = [x.count(Cnt.most_common(5)[4][0]) for x in l_less_sent_10 ]
# --------------------------------Q5-------------------------------------------------------------------------------------
count_len_sent = [len(x) for x in l_less_sent_10 ]
most_comon_work_in_sent = [Counter(x.split(' ')).most_common(5) for x in l_less_sent_10 ]

df = pd.DataFrame({'Sents':l_less_sent_10, 'Sent_len':count_len_sent, 'Number_comma':count_comma, "most_common_words":most_comon_work_in_sent,
                   'Number_quote ':count_quote ,'Number_comma':count_comma,'Number_prime':count_prime,
                   'Number_dash ':count_dash ,})

df.to_excel('Sample_Feature.xlsx')



