import re
urls = '''https://www.google.com
          http://amir.com
          https://youtube.com
          https://www.epa.gov
       '''
pattern1 = re.compile(r'https?://(www\.)?\w+\.\w+')
matches = pattern1.finditer(urls)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - url' + 20 * '-' )
pattern2 = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
matches = pattern2.finditer(urls)
for match in matches:
    print(match)
# ----------------------------
print(20 * '-' + 'Example - url' + 20 * '-')
pattern2 = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')
matches = pattern2.finditer(urls)
for match in matches:
    print(match.group(2))