# Titanic survival decision tree

EECE 5644, Assignment 04. This trains a decision tree to predict whether a Titanic passenger
survived (the `Survived` column, 0 = died, 1 = survived) from their passenger record. The tree is
tuned with a 5-fold grid search and then scored on a held-out test set, so the headline numbers are
on passengers the model never saw during training.

## What is in here

All files sit in the assignment-04 folder.

```
titanic_decision_tree.py   the script, steps 1 to 8, runs top to bottom
titanic.csv                the dataset (891 passengers)
data_dictionary.md         the twelve raw columns and the ten features the model uses
FINDINGS.md                what the model learned and how well it does
DECISIONS.md               the choices I made along the way
requirements.txt           pinned package versions
figures/
  decision_tree.png        the trained tree
  feature_importances.png  which features drove the predictions
```

## The data

titanic.csv is already in the folder (about 59 KB, 891 rows), so there is no download step. It is
the standard Kaggle Titanic training table with twelve columns. Three of them have missing values:
Cabin (687 missing, 77%), Age (177, 20%) and Embarked (2). The columns are described in
data_dictionary.md.

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python titanic_decision_tree.py
```

It reads titanic.csv from the same folder and writes both PNGs into figures/, so run it from the
assignment-04 directory. Every step prints its results, so you can follow along in the terminal.

## Main findings

The tuned tree (gini, max depth 6, minimum 5 samples per leaf) scores 0.82 accuracy in
cross-validation and 0.76 on the held-out test set. Sex is by far the strongest predictor. It alone
accounts for about 52% of the tree's decisions, and it is the first split. Passenger class (14%),
fare paid (11%) and age (10%) come next, all of them proxies for wealth, deck position and the
"women and children first" order of the evacuation.

The model is better at spotting passengers who died (recall 0.90) than passengers who survived
(recall 0.54). Of the 69 real survivors in the test set it catches 37 and misses 32. See FINDINGS.md
for the full write-up and DECISIONS.md for the reasoning.
