# Run Logs

## 2 million edges, 1023878 nodes

```
parsing indices
parsing arcs
done building graph
--- getting adjacency matrix
=== getting adjacency matrix takes 13.508808710001176s
--- computing sparse SVD
=== computing sparse SVD takes 1.5249451839990797s
--- computing spannogram
16350.809482488976
=== computing spannogram takes 2277.9820167200014s
metric 234.342355954
--- selecting subgraph
=== selecting subgraph takes 0.5553878800055827s
--- drawing subgraph
=== drawing subgraph takes 0.22923175599862589s
--- rendering
=== rendering takes 29.488394672996947s
```


## 10 kilo edges, 9530 nodes

```
parsing indices
parsing arcs
done building graph
--- getting adjacency matrix
=== getting adjacency matrix takes 0.03733408100379165s
--- computing sparse SVD
=== computing sparse SVD takes 0.013043289000052027s
--- computing spannogram
16350.809482488976
V.shape = (9530, 2)
[==================================================] 100%
~~~ dot takes 1.471585243307345s
~~~ argsort takes 9.89041278968216s
~~~ norm-scatter-square takes 0.9988982248614775s
=== computing spannogram takes 13.495791499000916s
metric 36.0377410571
--- selecting subgraph
=== selecting subgraph takes 0.003697252002893947s
--- drawing subgraph
=== drawing subgraph takes 0.22319087399955606s
--- rendering
=== rendering takes 3.554715131998819s
```

## 1 million edges, 475719 edges

```
parsing indices
parsing arcs
done building graph
--- getting adjacency matrix
=== getting adjacency matrix takes 5.606153391003318s
--- computing sparse SVD
=== computing sparse SVD takes 0.4875342340019415s
--- computing spannogram
16350.809482488976
V.shape = (475719, 2)
[==================================================] 100%
~~~ dot takes 72.87644639386417s
~~~ argsort takes 825.2065105106958s
~~~ norm-scatter-square takes 1.9766365759351174s
=== computing spannogram takes 902.6521226480036s
metric 334.853951623
--- selecting subgraph
=== selecting subgraph takes 0.2774148999960744s
--- drawing subgraph
=== drawing subgraph takes 0.6755617310045636s
--- rendering
=== rendering takes 43.91162431799603s
```


## with GPU iteration 1, 1 million, 475719 edges

```
parsing indices
parsing arcs
done building graph
--- getting adjacency matrix
=== getting adjacency matrix takes 5.680325224000001s
--- computing sparse SVD
=== computing sparse SVD takes 0.4968394879999778s
--- computing spannogram
16350.809482488976
V.shape = (475719, 2)
[==================================================] 100%
~~~ dot takes 66.26462529500503s
~~~ process takes 183.7300573470077s
=== computing spannogram takes 251.91745214500003s
metric 234.330151282
--- selecting subgraph
=== selecting subgraph takes 0.24239574400007768s
--- drawing subgraph
=== drawing subgraph takes 0.2091893470000059s
--- rendering
=== rendering takes 12.284106608999991s
```

## with GPU iteration 1, 1 million, 475719 edges

machine: nvidia1; k20; Intel(R) Core(TM) i7-4820K CPU @ 3.70GHz

GPU

```
parsing indices
parsing arcs
done building graph
--- getting adjacency matrix
=== getting adjacency matrix takes 4.886149043217301s
--- computing sparse SVD
=== computing sparse SVD takes 0.42436322197318077s
--- computing spannogram
16350.809482488976
V.shape = (475719, 2)
[==================================================] 100%
~~~ dot takes 29.794953227974474s
~~~ process takes 47.99520124681294s
=== computing spannogram takes 79.95194724900648s
metric 234.498037191
--- selecting subgraph
=== selecting subgraph takes 0.2196907983161509s
--- drawing subgraph
=== drawing subgraph takes 0.1772930920124054s
--- rendering
=== rendering takes 0.37241417821496725s
```

CPU
```
parsing indices
parsing arcs
done building graph
--- getting adjacency matrix
=== getting adjacency matrix takes 4.9144850466400385s
--- computing sparse SVD
=== computing sparse SVD takes 0.4155474817380309s
--- computing spannogram
16350.809482488976
V.shape = (475719, 2)
[==================================================] 100%
~~~ dot takes 32.09444721508771s
~~~ argsort takes 718.0756665794179s
~~~ norm-scatter-square takes 1.5464341761544347s
=== computing spannogram takes 754.2260599490255s
metric 234.364868276
--- selecting subgraph
=== selecting subgraph takes 0.21788877993822098s
--- drawing subgraph
=== drawing subgraph takes 0.17275578109547496s
--- rendering
=== rendering takes 0.35090011497959495s
```
