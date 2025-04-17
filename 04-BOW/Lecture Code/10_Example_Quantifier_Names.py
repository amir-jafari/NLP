import re
with open('re_text.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - Find Phone number' + 20 * '-' )
pattern1 = re.compile(r'\d{3}[-.]\d{3}[-.]\d{4}')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# # ----------------------------
print(20 * '-' + 'Example - Find name' + 20 * '-' )
pattern2 = re.compile(r'Mr\.?\s[A-Z]\w*')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # # ----------------------------
print(20 * '-' + 'Example - Find name with groups' + 20 * '-' )
pattern3 = re.compile(r'M(r|s|rs)\.?\s[A-Z]\w*')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# # # # ----------------------------
print(20 * '-' + 'Example - Find name with groups' + 20 * '-')
pattern4 = re.compile(r'(Mr|Ms|Mrs)\.?\s[A-Z]\w*')
matches = pattern4.finditer(text)
for match in matches:
    print(match)
