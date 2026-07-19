# ============================================
# House Price Predictor — ML Model
# Author: Sandun Godallage
# ============================================

import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

MODEL_PATH = "house_price_model.joblib"

# Model එකට input වෙන්න ඕනේ නිවැරදිම Feature Order එක
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

_HOUSING_CACHE = None


def create_data():
    """Real-world California Housing dataset load කරනවා"""
    global _HOUSING_CACHE
    if _HOUSING_CACHE is None:
        housing = fetch_california_housing(as_frame=False)
        data = np.column_stack([housing.data, housing.target])
        columns = list(housing.feature_names) + ["price"]
        df = pd.DataFrame(data, columns=columns)
        _HOUSING_CACHE = df
    return _HOUSING_CACHE


def train_model():
    """Model train කරනවා (Gradient Boosting)"""
    df = create_data()

    #FEATURES list එකේ තියෙන order එකටම X data ටික ගන්නවා
    X = df[FEATURES]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Accuracy check
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model trained! MAE: {round(mae, 4)} (${round(mae * 100000, 0):,.0f})")
    print(f"R2 Score: {round(r2, 4)}")

    return model


def save_model(model, path: str = MODEL_PATH):
    """Trained model eka disk ekata save කරනවා"""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH):
    """Disk eken model eka load කරනවා. File එක නැත්නම් auto-train කරනවා."""
    if not os.path.exists(path):
        print(f"Model file '{path}' not found. Training a new model first...")
        model = train_model()
        save_model(model, path)
        return model
    return joblib.load(path)


def predict_price(model, medinc, houseage, averooms, averageoccupation,
                  latitude, longitude, population, avebedrms):
    """Price predict කරනවා"""
    
    # පරාමිතීන් (arguments) pass කරපු order එකට නෙමෙයි, 
    # FEATURES list එකේ තියෙන order එකටම values ටික map කරගන්නවා.
    input_dict = {
        "MedInc": medinc, "HouseAge": houseage, "AveRooms": averooms,
        "AveOccup": averageoccupation, "Latitude": latitude, "Longitude": longitude,
        "Population": population, "AveBedrms": avebedrms
    }
    
    values = [input_dict[f] for f in FEATURES]

    for name, v in zip(FEATURES, values):
        if v is None:
            raise ValueError(f"{name} must be provided (got None)")
        try:
            float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number, got {v!r}")

    input_data = pd.DataFrame(
        [[float(v) for v in values]],
        columns=FEATURES,
        index=[0],
    )
    
    price = model.predict(input_data)
    return round(float(price[0]) * 100000, 2)


def feature_importance(model):
    """Feature importance (කොහොමද හෙද්දි price ලං කරයි)"""
    importances = model.feature_importances_
    table = (
        pd.DataFrame({"feature": model.feature_names_in_, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return table


def evaluate_model(model):
    """Test set metrics return කරනවා"""
    df = create_data()
    X = df[FEATURES]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    preds = model.predict(X_test)
    return {
        "mae": round(mean_absolute_error(y_test, preds) * 100000, 2),
        "r2": round(r2_score(y_test, preds), 4),
    }


def _run_cli():
    parser = argparse.ArgumentParser(description="House Price Predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train and save the model")

    pred = sub.add_parser("predict", help="Predict a price from features")
    pred.add_argument("--medinc", type=float, required=True)
    pred.add_argument("--houseage", type=float, required=True)
    pred.add_argument("--averooms", type=float, required=True)
    pred.add_argument("--averageoccupation", type=float, required=True)
    pred.add_argument("--latitude", type=float, required=True)
    pred.add_argument("--longitude", type=float, required=True)
    pred.add_argument("--population", type=float, required=True)
    pred.add_argument("--avebedrms", type=float, required=True)

    sub.add_parser("importance", help="Show feature importances")
    sub.add_parser("evaluate", help="Show test-set metrics")

    args = parser.parse_args()

    if args.command == "train":
        model = train_model()
        save_model(model)
    elif args.command == "predict":
        model = load_model()
        price = predict_price(
            model,
            medinc=args.medinc, houseage=args.houseage,
            averooms=args.averooms, averageoccupation=args.averageoccupation,
            latitude=args.latitude, longitude=args.longitude,
            population=args.population, avebedrms=args.avebedrms,
        )
        print(json.dumps({"predicted_price": price}, indent=2))
    elif args.command == "importance":
        model = load_model()
        print(feature_importance(model).to_string(index=False))
    elif args.command == "evaluate":
        model = load_model()
        print(json.dumps(evaluate_model(model), indent=2))


if __name__ == "__main__":
    _run_cli()