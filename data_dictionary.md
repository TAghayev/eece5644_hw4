# Data dictionary: titanic.csv

titanic.csv has 891 rows and 12 columns, one row per passenger. Survived is the target; the model
uses ten features built from the rest. PassengerId, Name and Ticket are dropped as identifiers, and
Cabin is dropped after HasCabin is engineered from it.

## Raw columns

| Column | Type | Missing | Meaning |
|--------|------|--------:|---------|
| PassengerId | int64 | 0 | Row identifier (dropped) |
| Survived | int64 | 0 | Target: 0 = died, 1 = survived |
| Pclass | int64 | 0 | Ticket class: 1 = first, 2 = second, 3 = third |
| Name | object | 0 | Passenger name (dropped) |
| Sex | object | 0 | male or female |
| Age | float64 | 177 | Age in years (fractional for infants) |
| SibSp | int64 | 0 | Siblings and spouses aboard |
| Parch | int64 | 0 | Parents and children aboard |
| Ticket | object | 0 | Ticket number (dropped) |
| Fare | float64 | 0 | Fare paid, in pounds |
| Cabin | object | 687 | Cabin number; dropped after HasCabin |
| Embarked | object | 2 | Port: S = Southampton, C = Cherbourg, Q = Queenstown |

## Features fed to the model

Ten features, in the order the script uses them.

| Feature | Type | Source | Values / meaning |
|---------|------|--------|------------------|
| Pclass | int64 | raw | 1, 2 or 3 |
| Sex | int64 | encoded | male = 0, female = 1 |
| Age | float64 | imputed | years; 177 missing filled with the median (28.0) |
| SibSp | int64 | raw | siblings/spouses aboard |
| Parch | int64 | raw | parents/children aboard |
| Fare | float64 | raw | fare paid, in pounds |
| Embarked | int64 | imputed + encoded | S = 0, C = 1, Q = 2; 2 missing filled with the mode (S) |
| HasCabin | int64 | engineered | 1 if a cabin was recorded, else 0 |
| FamilySize | int64 | engineered | SibSp + Parch + 1 |
| IsAlone | int64 | engineered | 1 if FamilySize == 1, else 0 |

## Notes

Age and Embarked are imputed on the full table before the split (see DECISIONS.md, point 4). Sex and
Embarked are the only text columns, and both are mapped to small integers; a decision tree treats
these as split thresholds, so the particular numbers do not matter. Fare has no missing values in
this file, so it needs no imputation.
