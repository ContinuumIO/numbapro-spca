from __future__ import division
import sys
import time


def render_progress(progress, width):
    length = round(max(min(progress, 1), 0) * width)
    sys.stdout.write('\r[')
    sys.stdout.write('=' * length)
    sys.stdout.write(' ' * (width - length))
    sys.stdout.write('] {percent}%'.format(percent=int(length / width * 100)))
    sys.stdout.flush()


def test():
    for i in range(100):
        render_progress(i / 100, 50)
        time.sleep(0.1)
