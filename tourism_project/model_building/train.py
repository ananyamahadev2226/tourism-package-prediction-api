import sys
from pathlib import Path
import pandas as pd
import joblib
import mlflow
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[2]

# 1. Load the train and test splits produced by the previous job
Xtrain = pd.read_csv(ROOT / "Xtrain.csv")
Xtest = pd.read_csv(ROOT / "Xtest.csv")
ytrain = pd.read_csv(ROOT / "ytrain.csv").squeeze("columns")
ytest = pd.read_csv(ROOT / "ytest.csv").squeeze("columns")

# Calculate class weights for scale_pos_weight if target is imbalanced
pos_count = (ytrain == 1).sum()
neg_count = (ytrain == 0).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

# 2. Define the model and a hyperparameter grid
numeric_features = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
                    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
                    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched",
                        "MaritalStatus", "Designation"]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss")
model = make_pipeline(preprocessor, xgb)

param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

# 3. Tune the model and log the parameters and metrics to MLflow
mlflow.set_experiment("wellness-tourism-package")

with mlflow.start_run():
    grid = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    # log every tuned parameter combination with its cross-validated F1
    for params, score in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1", score)

    best_model = grid.best_estimator_
    mlflow.log_params({f"best_{k}": v for k, v in grid.best_params_.items()})

    # 4. Evaluate the best model on the test set
    pred = best_model.predict(Xtest)
    mlflow.log_metrics({
        "test_accuracy": accuracy_score(ytest, pred),
        "test_precision": precision_score(ytest, pred),
        "test_recall": recall_score(ytest, pred),
        "test_f1": f1_score(ytest, pred),
    })
    print("Best parameters:", grid.best_params_)
    print(classification_report(ytest, pred))

# 5. Save the best model into tourism_project/deployment/ so the workflow can commit it
model_path = ROOT / "tourism_project" / "deployment" / "best_tourism_model.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, model_path)
print("Saved model to", model_path)
