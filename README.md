# Eurosat terrain classification
This repository is made for the Predictive Modeling course project at EPF. The goal of the project is to classify the Eurosat dataset using different machine learning algorithms. The dataset is available at https://www.tensorflow.org/datasets/catalog/eurosat

## Exploring the data
We can easily import the images with `tensorflow_datasets` and split them into train, validation and test sets. This gives us the following amounts:

```Python
Number of training samples:  16200
Number of validation samples:  5400
Number of test samples:  5400
```
