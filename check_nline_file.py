import sys

with open(sys.argv[1]) as f:
    count = 0
    for data in f:
        count += 1
        if count % 100000 == 0:
            print count
    print "Total line : "+str(count)