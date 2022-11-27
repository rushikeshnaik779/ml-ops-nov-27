#print("Hi This is rushikesh")
def add(s1, s2):
    s1 = list(s1)
    s2 = list(s2)

    if len(s1)>len(s2):
        length = len(s1)

    else: 
        length = len(s2)
    print(length)
    s = []
    for l in range(length):
        if len(s1)>l:
            s.append(s1[l])
        if len(s2)>l:
            s.append(s2[l])

    return ''.join(s)

print(add('Rushkesh', 'Sahil'))