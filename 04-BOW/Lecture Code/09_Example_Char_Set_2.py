import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - lower' + 20 * '-' )
pattern1 = re.compile(r'[a-z]')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - lower upper' + 20 * '-' )
pattern2 = re.compile(r'[a-zA-Z]')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - negate the lower and upper' + 20 * '-' )
pattern3 = re.compile(r'[^a-zA-Z]')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - Find words end with at' + 20 * '-' )
pattern1 = re.compile(r'[^b]at')
matches = pattern1.finditer(text)
for match in matches:
    print(match)