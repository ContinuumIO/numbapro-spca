"""
Build gzip files for each column

"""
# import zipfile
from collections import deque
import io
import gzip

index = 'pld-index'
# index = 'pld-index-1m.dat'
filename = 'pld-arc'
# filename = 'pld-arc-1m.dat'
outfmt = 'data/{0}'

start = 0
stop = 42889799 + 1
# stop = 100


class rewind_iterator(object):
    def __init__(self, iterator):
        self._iter = iter(iterator)
        self._buffer = deque()
        self._cur = None

    def __iter__(self):
        return self

    def __next__(self):

        if self._buffer:
            out = self._buffer.popleft()
        else:
            out = next(self._iter)

        self._cur = out
        return out

    def unread(self):
        assert self._cur is not None
        self._buffer.append(self._cur)
        self._cur = None


with open(filename) as fin:
    line_iter = iter(rewind_iterator(fin))
    with open(index) as findex:
        for indexline in findex:
            domain, node = indexline.split()
            node = int(node)
            print('node =', node)
            if node < start:
                continue
            elif node > stop:
                break

            ofile = outfmt.format(node)
            print('ofile', ofile)
            ct = 0
            with io.BytesIO() as fout:
                for line in line_iter:
                    a, b = map(int, line.split())
                    if a == node:
                        fout.write("{0}\n".format(b).encode('utf8'))
                        ct += 1
                    elif a > node:
                        line_iter.unread()
                        break
                fout.flush()
                if ct > 0:
                    print('ct =', ct)
                    with gzip.open(ofile, 'wb') as gzfile:
                        gzfile.write(fout.getvalue())

