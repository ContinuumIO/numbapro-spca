import sys
import zipfile

filename = sys.argv[1]

with zipfile.ZipFile(filename, compression=zipfile.ZIP_DEFLATED) as zipf:
    for name in zipf.namelist():
        print('==', name, '==')
        with zipf.open(name) as fin:
            for line in fin:
                print(int(line))
