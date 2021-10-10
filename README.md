# midi-shark

## Housekeeping Instructions: DEVELOPERS PLEASE READ
1. Please enter the **global** path that stores all your datasets into the `_ROOT` variable in `constants.py`. 
2. Please pair the names with individual paths of each datasets into the `_DATASETS` dict in `constants.py`, e.g. `"MAPS": "dir"`, where your dataset is stored in the directory `_ROOT/dir`.
3. This should be the **last** commit directly to the master branch. Please use proper github practice.
4. Please do **not** upload any raw data unless there is a **very good reason** (like as an example). This will clog up the repository.
5. For consistency, maintain the datasets in the same general file structure. If any changes to the file structure is deemed required, it must be communicated and approved by everybody on the team. For examples, scroll to the bottom.

## Resources/References

#### Datasets
- [MAESTRO](magenta.tensorflow.org/datasets/maestro)
- [MusicNet](homes.cs.washington.edu/~thickstn/start.html)
- [MAPS](www.telecom-paris.fr/fr/lecole/departements-enseignement-recherche/image-donnees-signal/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) currently unavailable

#### Papers
- [Onset Frames Paper](arxiv.org/abs/1710.11153)
- [Transformer Paper](arxiv.org/abs/1706.03762)
- [DeepLab V3 Paper](arxiv.org/abs/1802.02611)

## Dataset Default File Structures
If there are any new datasets added, please update the README with the file structures.
-  MAESTRO should look like
```
.
├── 2004
├── 2006
├── 2008
├── 2009
├── 2011
├── 2013
├── 2014
├── 2015
├── 2017
├── 2018
├── LICENSE
├── maestro-v3.0.0.csv
├── maestro-v3.0.0.json
└── README
```
- MusicNet should look like 
```
.
├── test_data
├── test_labels
├── train_data
└── train_labels
```
