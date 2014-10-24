import subprocess, itertools

fmt = 'python tallskinny.py {width} {size} {strategy}'

widths = [10000]
sizes = [10000, 20000, 30000]
strategies = ['mt', 'cpu', 'gpu']
for width, size, strategy in itertools.product(widths, sizes, strategies):
    if width <= size:
        print('=' * 80)
        print(width, size)
        out = subprocess.check_output(fmt.format(width=width,
                                                 size=size,
                                                 strategy=strategy),
                                      shell=True)
        print(out.decode('utf8'))
