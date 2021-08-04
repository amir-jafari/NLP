import re
with open('re_email.txt','r') as f:
    text = f.read()
# ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-' )
pattern1 = re.compile(r'[a-zA-z]+@[a-zA-z]+\.com')
matches = pattern1.finditer(text)
for match in matches:
    print(match)
# # # ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-' )
pattern2 = re.compile(r'[a-zA-z.]+@[a-zA-z]+\.(com|edu)')
matches = pattern2.finditer(text)
for match in matches:
    print(match)
# # # # # ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-' )
pattern3 = re.compile(r'[a-zA-z0-9.-]+@[a-zA-z-]+\.(com|edu|net)')
matches = pattern3.finditer(text)
for match in matches:
    print(match)
# # # # # ----------------------------
print(20 * '-' + 'Example - email' + 20 * '-')
pattern4 = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
matches = pattern4.finditer(text)
for match in matches:
    print(match)
