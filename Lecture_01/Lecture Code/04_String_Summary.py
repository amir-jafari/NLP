S1= '1'
S2 = '1abc'
S3 = 'Acd'
S4 = 'abcd. Abcd.'
S5 = '     abcd'
S6 = ' '
print(S4.capitalize())
print(S4.count('c'))
print(S1.isdigit())
print(S2.isdigit())
print(S4.isalnum())
print(S3.encode('UTF-8'))
print(S3.encode('UTF-16'))
print(S1.center(4))
print(S5.strip())
print(S4.index('c'))
print(S6.isspace())
print(S3.istitle())
print('.'.join(S3))
print(S4.split(sep='.'))
