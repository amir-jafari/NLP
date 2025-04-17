import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Word' + 20 * '-' )
pattern1 = re.compile(r'\w')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Word' + 20 * '-' )
pattern2 = re.compile(r'\W')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Word' + 20 * '-' )
pattern3 = re.compile(r'\w\.\w')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
