## Subsampled Dataset

Subsampled datasets are created from the first N arcs.  We then create the
index file from the full index with the nodes appeared in the subsampled arc
file.

```bash
curl http://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/pld-arc | head -n 10000 > pld-arc-10k
python wdc_dataset.py pld-arc-10k pld-index pld-index-10k
```

## Download

From http://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/

