"""
Build tall skinny matrix
"""

index = 'data/pld-index'
filename = 'pld-arc'
output = 'data/arc-%d.dat'

start = 0
stop = 10000000

with open(index) as findex:
    for indexline in findex:
        domain, node = indexline.split()
        node = int(node)
        print('node =', node)
        if node < start:
            continue
        if node > stop:
            break
        with open(filename) as fin:
            ofile = output % node
            print(ofile)
            with open(ofile, 'w') as fout:
                ct = 0
                for line in fin:
                    a, b = map(int, line.split())
                    if a == node:
                        ct += 1
                        print(b, file=fout)
                    elif a > node:
                        break
                print('count', ct)
