import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern1 = re.compile(r'\d\d\d[-.]\d\d\d[-.]\d\d\d\d')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern2 = re.compile(r'[89]00[-.]\d\d\d[-.]\d\d\d\d')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern3 = re.compile(r'[1-5]')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
