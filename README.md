# TFG_Yixi
## Version 1.0.
MPNNs are build with chemprop to predict metabolites RT (retention time.)

Inside src/training/RepoRT/... are found many "main.py", just run those files, it will work. Only if it were the first time running, it will take several minutes building the input datafiles, but once built, it won't build them again, unless input pathes are changed. The raw data:
```
https://github.com/michaelwitting/RepoRT
```
Inside src/training/SMRT/... are found 3 scripts, works in the same way as described before. Only if SMRT datafile were not found (should not be the case if this repo is cloned), you should download the original dataset from:
```
https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913
```
SMRT data and its molecular descriptors come with this repo, as it would take very long to build them.

References:
```angular2html
Esther Heid, Kevin P. Greenman, Yunsie Chung, Shih-Cheng Li, David E. Graff, Florence H. Vermeire, Haoyang Wu, William H. Green, and Charles J. McGill
Journal of Chemical Information and Modeling 2024 64 (1), 9-17
DOI: 10.1021/acs.jcim.3c01250 (chemprop)

Domingo-Almenara, X., Guijas, C., Billings, E. et al. 
The METLIN small molecule dataset for machine learning-based retention time prediction. Nat Commun 10, 5811 (2019). 
https://doi.org/10.1038/s41467-019-13680-7 (SMRT)

Kretschmer, F., Harrieder, EM., Hoffmann, M.A. et al. 
RepoRT: a comprehensive repository for small molecule retention times. Nat Methods 21, 153–155 (2024). 
https://doi.org/10.1038/s41592-023-02143-z (RepoRT)

```