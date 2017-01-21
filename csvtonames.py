f = open("signnames.csv","r")

id = 0
for x in f.readlines():
    r = x.split(",")
    print('"{}",'.format(r[1][:-1]))