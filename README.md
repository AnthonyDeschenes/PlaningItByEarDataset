# PlaningItByEarDataset
This project present the dataset used in our article Planing it by ear: Convolutional neural networks for acoustic anomaly detection in industrial wood planers submitted to ICASSP 2025. Note that in the article no seeds were set, so the results might be different from the published results. To ensure better reproducibility we set the seeds in this repository.

# Installation
- Simply clone the project and install the package in the requirements.txt. The experiments where conducted with python 3.10.
- Unzip the data.zip file. The data folder should be on the same level as the models and trainer repositories.

# Requirements
- 16GB of RAM
- Intel Core i7

# Run
Run the file train_and_evaluate.py
- This will train the models on the training set and produce .joblib files, which are the stored models.

Run on the evaluation dataset
- Change the boolean at line 29 of train_and_evaluate.py to False.
- Run the file again.
- This will produce the ROC curve of the article.

# Citation
TODO
