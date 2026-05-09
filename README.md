# 🎯 Prediction of FIFA Football Players' Potential and Position Type

A Machine Learning project applying regression and classification techniques to predict two key aspects of FIFA football players based on their physical attributes, skills, and performance statistics.

---

## 📌 Project Overview

| | |
|---|---|
| **Regression Task** | Predict players' **potential rating** (continuous variable) |
| **Classification Task** | Predict players' **position type** (Goalkeeper / Defender / Midfielder / Attacker) |
| **Dataset** | [Kaggle — FIFA Players Dataset](https://www.kaggle.com/datasets/maso0dahmed/football-players-data/data) |
| **Original Source** | [SoFIFA.com](https://sofifa.com/) |
| **Sample Size** | 17,954 players × 51 features |

---

## 🗂️ Table of Contents

- [Data Overview](#-data-overview)
- [Feature Engineering](#-feature-engineering)
- [Project Structure](#-project-structure)
- [Part I — Regression](#-part-i--regression-predicting-player-potential)
- [Part II — Classification](#-part-ii--classification-predicting-position-type)
- [Final Results](#-final-results)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)

---

## 📦 Data Overview

**Source**: [Kaggle — Football Players Dataset](https://www.kaggle.com/datasets/maso0dahmed/football-players-data/data), originally scraped from [SoFIFA.com](https://sofifa.com/)

**Key variables**: player demographics, technical and mental skills, physical metrics (height, weight, BMI), and gameplay attributes.

**Position Classification Used:**

| Category | Positions |
|----------|-----------|
| 🧤 Goalkeepers | GK |
| 🛡️ Defenders | CB, LB, RB, LWB, RWB |
| ⚙️ Midfielders | CM, CAM, CDM, LM, RM |
| 🎯 Attackers | ST, CF, LW, RW, LS, RS |

---

## ⚙️ Feature Engineering

New features created from raw data:

| Feature | Description |
|---------|-------------|
| `attacking`, `skill`, `movement`, `power`, `mentality`, `defending` | Skill group averages (6 composite features) |
| `bmi` | Body Mass Index from height and weight |
| `is_tall` | Binary flag: height > 185 cm |
| `is_heavy` | Binary flag: weight > 80 kg |
| `continent` | Player's nationality mapped to continent |
| `eu_player` | Binary flag for European players |
| `age_group` | Categorical: <20 / 20–25 / 26–30 / 31–35 / 35+ |
| `major_position` | Primary position extracted from positions list |
| `position_type` | Grouped role: Goalkeeper / Defender / Midfielder / Attacker |

---

## 📁 Project Structure

```
football-players-ml/
│
├── ML_Project_on_Football_Players_Data.ipynb   # Main notebook
│
├── data/
│   └── fifa_players.csv                        # Raw dataset (from Kaggle)
│
└── README.md
```

---

## 🌟 Part I — Regression: Predicting Player Potential

**Target**: `potential` (continuous, range ~45–95)

**Models tested**: Ridge · Lasso · SGD · Decision Tree · Random Forest · XGBoost

**Preprocessing pipeline**: StandardScaler + SimpleImputer for numeric features, OneHotEncoder for categorical features, GridSearchCV / Optuna for hyperparameter tuning.

### 📊 Regression Results

| Model | R² Test | RMSE Test | MAPE Test | CV RMSE | Notes |
|-------|---------|-----------|-----------|---------|-------|
| Ridge Regression | 0.8477 | 2.00 | 2.43% | — | Stable, no features removed |
| Lasso Regression | 0.8477 | 2.00 | 2.42% | — | 12 features zeroed out |
| SGD Regression | 0.8435 | 2.00 | 2.39% | 2.46 | Competitive with linear models |
| Decision Tree | 0.9497 | 1.00 | 1.10% | 1.53 | ⚠️ Overfitting detected |
| Random Forest | 0.9667 | 1.00 | 0.94% | 1.24 | Strong generalization |
| **XGBoost** | **0.9687** | **1.085** | **1.02%** | **1.16** | ✅ **Best overall** |

### 🏆 Best Regression Model: XGBoost
- **R² = 0.9687** — explains nearly 97% of potential's variance
- **CV RMSE = 1.16** — best generalization score across all models
- Built-in regularization effectively controls overfitting

### 🔑 Key Predictors of Potential
1. **Age group** — Players under 20 receive the largest potential boost (+9–13 points)
2. **Overall rating** — Current ability is the strongest numerical predictor (+6 points)
3. **South American origin** — Slight positive regional effect

---

## 🌟 Part II — Classification: Predicting Position Type

**Target**: `position_type` (4 classes: Goalkeepers / Defenders / Midfielders / Attackers)

**Models tested**: Logistic Regression · Random Forest · XGBoost · CatBoost

**Note**: Class distribution is moderately imbalanced; `class_weight='balanced'` / `auto_class_weights='Balanced'` applied where applicable.

### 📊 Classification Results

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|-------|----------|----------|-------------|---------|
| Logistic Regression | 84.69% | 0.87 | 0.85 | — |
| Random Forest | 84.80% | 0.87 | 0.85 | — |
| XGBoost | 85.37% | 0.87 | 0.85 | — |
| **CatBoost** | **85.63%** | **0.88** | **0.86** | **0.972** |

### 📋 CatBoost Class-wise Performance (Best Model)

| Position | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 🧤 Goalkeepers | 1.00 | 1.00 | 1.00 |
| 🛡️ Defenders | 0.88 | 0.91 | 0.89 |
| 🎯 Attackers | 0.78 | 0.85 | 0.81 |
| ⚙️ Midfielders | 0.83 | 0.77 | 0.80 |

### 🏆 Best Classification Model: CatBoost
- **Accuracy = 85.63%** — highest overall
- **ROC-AUC = 0.972** — superior class separation
- **Native categorical feature handling** — no manual encoding needed
- Best attacker recall (85%) among all models

---

## 💡 Final Results

### Regression Winner: XGBoost 🥇
> Best generalization (CV RMSE = 1.16), highest R² (0.9687), minimal overfitting

### Classification Winner: CatBoost 🥇
> Best accuracy (85.63%), highest ROC-AUC (0.972), excellent across all positions

---

## ⚙️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
optuna
statsmodels
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost optuna statsmodels
```

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/football-players-ml.git
   cd football-players-ml
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/maso0dahmed/football-players-data/data) and place `fifa_players.csv` in the `data/` folder.

3. Update the file path in the notebook (Cell 1):
   ```python
   # Change this line:
   df = pd.read_csv(r'C:\Users\User\Desktop\...\fifa_players.csv')
   
   # To:
   df = pd.read_csv('data/fifa_players.csv')
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook ML_Project_on_Football_Players_Data.ipynb
   ```

5. Run all cells sequentially. Optuna hyperparameter tuning may take several minutes.

---

## 📝 Notes

- Hyperparameter tuning uses **Optuna** (Bayesian optimization) for tree-based models and **GridSearchCV** for linear models.
- **VIF analysis** revealed severe multicollinearity among skill group features (VIF > 100), addressed via Ridge/Lasso regularization.
- Body type entries for famous players (Messi, Ronaldo, Neymar, etc.) were standardized to `'Unique'` during data cleaning.
- The `Akinfenwa` body type entry was removed as it represented a single non-standard observation.

---

*Data Science course project | Machine Learning module*
