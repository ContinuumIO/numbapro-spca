This directory contains all benchmark results.
A benchmark data file is a text file with the following formatting:
* each row is line delimited
* each column is comma delimited
* spaces are ignored

Each row should contain columns in the following order:
* k
* d
* GPU time
* CPU time
* GPU result
* CPU result

# File Naming Convention

`Name_GPUmodel_Precision.txt`

# Plotting

Use the `plotter.py` to plot:

```bash
    $ python plotter.py benchmark/bm1_gtx780ti_f64.txt
```
