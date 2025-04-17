import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Word Boundry' + 20 * '-' )
pattern1 = re.compile(r'\bHa')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern2 = re.compile(r'\BHa')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern3 = re.compile(r'\b\s\(N')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
