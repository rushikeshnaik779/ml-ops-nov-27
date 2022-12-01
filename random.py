def inf_sequence():
    num = 0
    while True:
        yield num
        num += 1
        if num == 10000:
            return 0  
         
for i in inf_sequence():
    print(i, end=" ")