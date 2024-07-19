# Group 1 Final Project
This project is a part of the AAI-511 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**-- Project Status: Active**

- ### Partner(s)/Contributor(s)
   * Dominic Fanucchi
   * Gabriel Emanuel Colón
   * Parker Christenson

## Installation
To use this project, first clone the repo on your device using the command below:
```bash
git init
git clone https://github.com/dominicfanucchi/aai-511_group1.git
```

## Project Objective
*Music Genre and Composer Classification Using Deep Learning*  
The primary objective of this project is to develop a deep learning model that can predict the composer of a given musical score accurately. The project aims to accomplish this objective by using two deep learning techniques: Long Short-Term Memory (LSTM) and Convolutional Neural Network (CNN).

## About the Dataset
The dataset contains the midi files of compositions from well-known classical composers like Bach, Beethoven, Chopin, and Mozart. The MIDI datset was sourced from this Kaggle [dataset](https://www.kaggle.com/datasets/blanderbuss/midi-classic-music) in which we removed all composers except for Bach, Beethoven, Chopin, and Mozart.

## Approach
The specific algorithms and networks used were as follows: 


These algorithms and networks were implemented through Python and Jupyter Notebooks. 

### Imports and Libraries
The project relies on the following libraries and packages:
* `os` - Provides a way of using operating system-dependent functionality.
* `csv` - Implements classes to read and write tabular data in CSV format.
* `warnings` - Provides a way to issue warnings and control their behavior.
* `numpy` - Fundamental package for numerical computations in Python.
* `pandas` - Data manipulation and analysis library.
* `matplotlib.pyplot` - Plotting library for creating static, interactive, and animated visualizations.
* `seaborn` - Statistical data visualization library based on Matplotlib.
* `%matplotlib inline` - Jupyter magic command for displaying plots inline within Jupyter notebooks.
* `pretty_midi` - Library for handling and analyzing MIDI files.
* `mido` - Library for working with MIDI files and ports.
* `sklearn.preprocessing` - Tools for preprocessing data, including `StandardScaler`, `LabelEncoder`, and `label_binarize`.
* `sklearn.model_selection` - Functions for splitting data and model selection, including `train_test_split`.
* `sklearn.metrics` - Metrics for evaluating models, including `confusion_matrix`, `classification_report`, `roc_curve`, and `auc`.
* `tensorflow` - Open-source platform for machine learning.
* `tensorflow.keras` - High-level neural networks API, including `Model`, `Input`, `LSTM`, `Conv2D`, `MaxPooling2D`, `Dense`, `Flatten`, and `Concatenate`.
* `torch` - PyTorch library for deep learning.
* `tqdm` - Library for showing progress bars in loops.

## Results
The proposed project aims to use deep learning techniques to accurately predict the composer of a given musical score. The project will be implemented using LSTM and CNN architectures and will involve data pre-processing, feature extraction, model building, training, and evaluation. The final model can be used by novice musicians, listeners, and music enthusiasts to identify the composer of a musical piece accurately.

## References

## Acknowledgments
We would like to express our sincere gratitude to... 

## License
This dataset is licensed under a [CC0 1.0 DEED license](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) - see the [Creative Commons](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) website for details.
