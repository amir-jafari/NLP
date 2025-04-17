import re
with open('re_text.txt','r') as f:
    text = f.read()
with open('re_fake_names.txt','r') as f:
    text1 = f.read()
# ----------------------------
print(20 * '-' + 'Example - Find Phone number 1' + 20 * '-' )
pattern1 = re.compile(r'\d\d\d')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - Find Phone number 2' + 20 * '-' )
pattern2 = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - Find Phone number 3' + 20 * '-' )
pattern3 = re.compile(r'\d\d\d.\d\d\d.\d\d\d\d')
matches = pattern3.finditer(text1)
for match in matches:
    print(match)
