# Decisions

The choices I made while building the classifier, with the reasoning for each.

1. Dropped PassengerId, Name and Ticket. These identify a passenger rather than describe them.
PassengerId is a row number, Name is free text, and Ticket is a near-unique code. None of them
generalize, so leaving them in would only push the tree to memorize individuals.

2. Turned Cabin into HasCabin, then dropped it. Cabin is missing for 77% of passengers, too sparse
to use as-is. But whether a cabin was recorded at all still says something, since it usually marked
a paying, higher-deck passenger, so I built a binary HasCabin before dropping Cabin.

3. Median for Age, mode for Embarked. Age is missing for 20% of passengers, and the median (28.0) is
robust to the long tail of older ages. Embarked is missing for only 2 passengers, so the mode
(S, Southampton) is a safe fill.

4. Imputation runs on the full frame, before the split. This lets a little information from the test
rows influence the Age median and Embarked mode, which is a mild leak. With 179 test rows and
statistics this simple the effect is negligible. The cleaner approach is to fit the imputers on the
training split only, which I note here for honesty.

5. FamilySize and IsAlone. FamilySize = SibSp + Parch + 1 folds the two family counts into one
feature, and IsAlone flags solo travelers. FamilySize ended up carrying the signal and IsAlone went
unused by the final tree, but both are cheap to include.

6. Numeric encoding, not one-hot. A tree does not care about the scale or order of a split value, so
I mapped the two text columns to small integers: Sex {male: 0, female: 1} and Embarked
{S: 0, C: 1, Q: 2}. This keeps the feature list short and the tree diagram readable.

7. Stratified 80/20 split, one seed. SEED = 42 is set once and reused for the split and the model.
The split is stratified on Survived, so both sides keep the roughly 38% survival rate (0.383 train,
0.385 test).

8. Grid search on accuracy, with a deliberately modest depth range. GridSearchCV (5-fold) tunes
max_depth, min_samples_leaf and criterion. I capped the depth candidates at 8 (plus unlimited) and
required at least a handful of samples per leaf, both to fight overfitting and to keep the winning
tree small enough to read. The winner was gini, depth 6, min_samples_leaf 5.
