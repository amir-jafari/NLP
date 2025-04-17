import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Digits' + 20 * '-' )
pattern1 = re.compile(r'\d')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Digits' + 20 * '-' )
pattern2 = re.compile(r'\D')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Digits' + 20 * '-' )
pattern3 = re.compile(r'\d\d')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
