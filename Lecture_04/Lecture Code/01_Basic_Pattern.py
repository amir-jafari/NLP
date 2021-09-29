import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'search a specific  Char' + 20 * '-' )
pattern1 = re.compile(r'abc')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
print(text[0:3])
# ----------------------------
print(20 * '-' + 'dot menas any chars except new line' + 20 * '-' )
pattern2 = re.compile(r'.')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Back slash escapes special characters' + 20 * '-' )
pattern3 = re.compile(r'\.')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Search an string' + 20 * '-' )
pattern4 = re.compile(r'amir\.com')
matches = pattern4.finditer(text)
for match in matches:
    print(match)





