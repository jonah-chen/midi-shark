# midi-shark

[![Presentation](https://i9.ytimg.com/vi/QNVYxn5-YnY/sddefault.jpg?v=61a7def9&sqp=CPyt544G&rs=AOn4CLAQuk5eZGMuc0cnjdopW6m09zIJTg)](https://youtu.be/QNVYxn5-YnY)

## Usage:

Make sure that requirements in `requirements.txt` are installed, or run
```
pip install -r requirements.txt
```
Then, make sure you have [FluidSynth](https://www.fluidsynth.org/) and a `.sf2` soundfont installed.

#### Preprocessing the Data. 
1. Create a file named `.env` in the project's root directory, following the template shown in the `.env.example` file.
2. Execute `processing/preprocess_batch.py` using Python. You must have the dataset and sufficient disk space of [] MB to store the preprocessed data. If you wish to only preprocess a subset, specify the `--year` argument.

#### Training a Model
1. Dataloaders (for pytorch) for all components of the dataset is located in [model/database.py](https://github.com/jonah-chen/midi-shark/blob/17c212d4d3ec920e250edcbe3f6f803a324ade95/model/database.py). Use this to load your data.
2. Then, you can train the models we have built using the `fit` method, and evaluate them using the `val_split` method. To use your own models, you can still use the dataloaders.

#### Making Predictions
1. You can use the code in [this jupyter notebook](https://github.com/jonah-chen/midi-shark/blob/17c212d4d3ec920e250edcbe3f6f803a324ade95/pred.ipynb) to make predictions. However, ensure you have trained some sort of model to make the predictions.

## Resources/References

#### Datasets
- [MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
- [MusicNet](https://homes.cs.washington.edu/~thickstn/start.html)
- [MAPS](https://www.telecom-paris.fr/fr/lecole/departements-enseignement-recherche/image-donnees-signal/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) currently unavailable

#### Papers
- [Onset Frames Paper](https://arxiv.org/abs/1710.11153)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)
- [DeepLab V3 Paper](https://arxiv.org/abs/1802.02611)

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
