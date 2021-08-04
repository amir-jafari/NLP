import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern1 = re.compile(r'\s')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern2 = re.compile(r'\S')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Space' + 20 * '-' )
pattern3 = re.compile(r'\s\.\s')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
