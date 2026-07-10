from pathlib import Path

import matplotlib

matplotlib.use("Agg")  

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
DATA_PATH = "titanic.csv"
TARGET = "Survived"

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "HasCabin",
    "FamilySize",
    "IsAlone",
]

DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

SEX_MAP = {"male": 0, "female": 1}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}

CLASS_NAMES = ["Died", "Survived"]

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 110

COLOR_BAR = "#2c7fb8"
COLOR_ACCENT = "#c0392b"

def step(number, title):
    """Print a numbered step banner so progress is easy to follow."""
    line = "=" * 74
    print(f"\n{line}\nStep {number}: {title}\n{line}")


def missing_summary(frame):
    """Return a table of columns that have missing values, with counts and %."""
    counts = frame.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    pct = (counts / len(frame) * 100).round(1)
    return pd.DataFrame({"missing": counts, "percent": pct})


def classification_metrics(y_true, y_pred):
    """Return accuracy, precision, recall and F1 for the survived (1) class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def load_data():
    """Step 1: read the CSV and report shape, dtypes and missing values."""
    step(1, "Load and inspect")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {DATA_PATH}: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("Column dtypes:")
    print(df.dtypes.to_string())

    print("\nMissing-value summary (columns with any missing):")
    print(missing_summary(df).to_string())
    return df


def preprocess(df):
    """Step 2: engineer features, impute, encode, and select the feature matrix."""
    step(2, "Preprocess")
    data = df.copy() 
    
    data["HasCabin"] = data["Cabin"].notna().astype(int)
    print("Engineered HasCabin from Cabin (1 = cabin recorded, 0 = missing).")

    data = data.drop(columns=DROP_COLS)
    print(f"Dropped columns: {', '.join(DROP_COLS)}")

    age_median = data["Age"].median()
    data["Age"] = data["Age"].fillna(age_median)
    embarked_mode = data["Embarked"].mode()[0]
    data["Embarked"] = data["Embarked"].fillna(embarked_mode)
    print(f"Imputed Age with median ({age_median:.1f}) and "
          f"Embarked with mode ('{embarked_mode}').")

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)
    print("Engineered FamilySize (SibSp + Parch + 1) and IsAlone (FamilySize == 1).")

    data["Sex"] = data["Sex"].map(SEX_MAP)
    data["Embarked"] = data["Embarked"].map(EMBARKED_MAP)
    print(f"Encoded Sex {SEX_MAP} and Embarked {EMBARKED_MAP}.")

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    print(f"\nFinal feature list ({len(FEATURES)}): {', '.join(FEATURES)}")
    return X, y


def split_data(X, y):
    """Step 3: stratified 80/20 train/test split."""
    step(3, "Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows  "
          f"(80/20, stratified on {TARGET}, seed {SEED})")
    print(f"Survival rate: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def tune_model(X_train, y_train):
    """Step 4: grid-search a decision tree over depth, leaf size and criterion."""
    step(4, "Train and tune (GridSearchCV, 5-fold)")
    param_grid = {
        "max_depth": [3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    }
    base = DecisionTreeClassifier(random_state=SEED)
    search = GridSearchCV(
        base, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    print(f"Searching {np.prod([len(v) for v in param_grid.values()])} "
          f"combinations across 5 folds...")
    search.fit(X_train, y_train)

    print("\nBest parameters:")
    for name, value in search.best_params_.items():
        print(f"  {name:16s}= {value}")
    print(f"Best 5-fold CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate(model, X_test, y_test):
    """Step 5: score the tuned tree on the held-out test set."""
    step(5, "Evaluate on the test set")
    y_pred = model.predict(X_test)

    metrics = classification_metrics(y_test, y_pred)
    print("Test-set metrics (positive class = Survived):")
    for name, value in metrics.items():
        print(f"  {name.capitalize():10s}: {value:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"                 pred Died  pred Survived")
    print(f"  actual Died   {cm[0, 0]:10d}  {cm[0, 1]:13d}")
    print(f"  actual Survived {cm[1, 0]:8d}  {cm[1, 1]:13d}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    return metrics


def feature_importances(model):
    """Step 6: rank the tree's feature importances high to low."""
    step(6, "Feature importances")
    importances = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )
    print("Feature importances (high to low):")
    for name, value in importances.items():
        print(f"  {name:12s} {value:.4f}")
    return importances


def make_figures(model, importances, test_accuracy):
    """Step 7: save the tree diagram and the feature-importance bar chart."""
    step(7, "Save figures")

    fig, ax = plt.subplots(figsize=(24, 13))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(
        f"Titanic survival decision tree (depth {model.get_depth()}, "
        f"{model.get_n_leaves()} leaves)",
        fontsize=16,
    )
    tree_path = FIG_DIR / "decision_tree.png"
    fig.savefig(tree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {tree_path}")

    ordered = importances.sort_values()  
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(ordered.index, ordered.values, color=COLOR_BAR)
    for bar, value in zip(bars, ordered.values):
        ax.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color=COLOR_ACCENT,
        )
    ax.set_xlabel("Importance (Gini/entropy reduction)")
    ax.set_xlim(0, ordered.max() * 1.15)
    ax.set_title(
        f"What drives the survival prediction\n"
        f"decision tree, test accuracy = {test_accuracy:.3f}"
    )
    fig.tight_layout()
    imp_path = FIG_DIR / "feature_importances.png"
    fig.savefig(imp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {imp_path}")


def print_summary(importances, metrics):
    """Step 8: plain-language read-out of what drove the predictions."""
    step(8, "Plain-language summary")
    top = importances.head(3)
    top_names = list(top.index)
    label = {
        "Sex": "sex",
        "Fare": "fare paid",
        "Pclass": "passenger class",
        "Age": "age",
        "FamilySize": "family size",
        "HasCabin": "having a recorded cabin",
        "IsAlone": "traveling alone",
        "Embarked": "port of embarkation",
        "SibSp": "siblings/spouse aboard",
        "Parch": "parents/children aboard",
    }
    readable = [label.get(name, name) for name in top_names]

    print(
        f"The tree predicts survival with about {metrics['accuracy'] * 100:.1f}% "
        f"accuracy on unseen passengers.\n"
        f"It leans mostly on {readable[0]} (importance {top.iloc[0]:.2f}), then "
        f"{readable[1]} ({top.iloc[1]:.2f}) and {readable[2]} ({top.iloc[2]:.2f}).\n"
        f"So {readable[0]} matters most, while fare and class stand in for how\n"
        f"wealthy a passenger was and where their cabin sat on the ship. That lines\n"
        f"up with the 'women and higher-class passengers first' order of the evacuation."
    )


def main():
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = tune_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    importances = feature_importances(model)
    make_figures(model, importances, metrics["accuracy"])
    print_summary(importances, metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
