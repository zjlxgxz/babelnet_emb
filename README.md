# change path 

## Python 3.7/8 install numba/scipy/dataclasses/...

## Data folder structure:
```
(base) xingzguo@pascal:~/projects_data$ tree -h --filelimit=5  babelnet_emb/
babelnet_emb/
├── [1.1G]  graph_embeddings.zip
├── [4.0K]  graph_query
│   ├── [ 20K]  framenet [260 entries exceeds filelimit, not opening dir]
│   └── [ 20K]  kalm [250 entries exceeds filelimit, not opening dir]
├── [190K]  graph_query_v2.zip
├── [292M]  synsets.csv
└── [6.0G]  triples.csv
```

## Command examples:
```
    # command example [high-precision, slow ]
    # python main.py --ppr_epsilon 1e-10  --ppr_alpha 0.2

    # command example [low-precision, fast ]
    # python main.py --ppr_epsilon 1e-2  --ppr_alpha 0.2

    # command example [low-precision, faster ]
    # python main.py --ppr_epsilon 1e-2  --ppr_alpha 0.8
```

The results (pandas dataframe) will be store in the ```graph_query``` folder with same file name but with ```.plk``` extension where each entry has the following structure:

```
    record = {
        "u": u, # start node id
        "v": v, # end node id
        "emb_u": emb_u, # node embeddings
        "emb_v": emb_v, # ..
        "proc_time": proc_time, # elapsed time for this file (not this pair)
        "l2-dist": distance.euclidean(emb_u, emb_v), 
        "cosine-dist": distance.cosine(emb_u, emb_v),
    }

```