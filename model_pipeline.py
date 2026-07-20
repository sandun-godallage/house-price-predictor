# =====================================================================
# Industrial House Price Predictor — Enterprise ML Pipeline
# Author: Sandun Godallage (Refactored to Production-Grade Pipeline)
# =====================================================================

import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing

MODEL_PATH = "industrial_house_model.joblib"

#   Feature  (Order)
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

_HOUSING_CACHE = None


def create_data():
    """Real-world California Housing dataset එක memory cache එකක් සහිතව load කරයි"""
    global _HOUSING_CACHE
    if _HOUSING_CACHE is None:
        # Dataset  Pandas DataFrame 
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.copy()
        df = df.rename(columns={"MedHouseVal": "price"})
        _HOUSING_CACHE = df
    return _HOUSING_CACHE


def build_industrial_pipeline():
    """Data ordering කතුර සහ AI මොළය එකට එකලස් කර Pipeline එක සාදයි"""
    # 1. Preprocessor: , FEATURES  Auto 
    #  Columns  Auto Drop (Drop remainder) 
    preprocessor = ColumnTransformer(
        transformers=[
            ("feature_order_lock", "passthrough", FEATURES)
        ],
        remainder="drop"
    )

    # 2. Complete Pipeline Assembly
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ))
    ])
    return pipeline


def train_model():
    """මුළු Pipeline එකම එකවර Train කර Accuracy පරීක්ෂා කරයි"""
    df = create_data()

    # Raw Data  X  Y  (Pipeline order )
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    pipeline = build_industrial_pipeline()
    
    print("Training the Industrial Beast Pipeline...")
    #  Data Ordering  Model Training !
    pipeline.fit(X_train, y_train)

    # Accuracy Checks
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n[+] Model Trained Successfully!")
    print(f"    - MAE: {round(mae, 4)} (${round(mae * 100000, 0):,.0f})")
    print(f"    - R2 Score: {round(r2, 4)}")

    return pipeline


def save_model(model, path: str = MODEL_PATH):
    """මුළු Pipeline එකම එක ගුලියක් ලෙස Disk එකට Save කරයි"""
    joblib.dump(model, path)
    print(f"[+] Pipeline saved safely to '{path}'")


def load_model(path: str = MODEL_PATH):
    """Disk එකෙන් Pipeline එක load කරයි. නැත්නම් Auto-Train වෙයි."""
    if not os.path.exists(path):
        print(f"[-] Model file '{path}' not found. Initializing Auto-Train...")
        model = train_model()
        save_model(model, path)
        return model
    return joblib.load(path)


def predict_price(pipeline, medinc, houseage, averooms, averageoccupation,
                  latitude, longitude, population, avebedrms):
    """ඕනෑම පිළිවෙලකට එන දත්ත Pipeline එක හරහා ගලපා නිවාස මිල ගණනය කරයි"""
    
    # ලැබෙන දත්ත Dictionary එකකට දමා DataFrame එකක් සාදයි
    raw_input = pd.DataFrame([{
        "MedInc": medinc, "HouseAge": houseage, "AveRooms": averooms,
        "AveOccup": averageoccupation, "Latitude": latitude, "Longitude": longitude,
        "Population": population, "AveBedrms": avebedrms
    }])

    # කිසිදු manual ordering එකක් අවශ්‍ය නැත; Pipeline එක විසින් එය බලාගනී!
    price = pipeline.predict(raw_input)
    return round(float(price[0]) * 100000, 2)


def feature_importance(pipeline):
    """මොඩල් එකේ මිල තීරණය කිරීමට වැඩිපුරම බලපෑ සාධක පෙන්වයි"""
    # Pipeline එක ඇතුළෙන් Regressor එක සහ Preprocessor එක වෙනම ගලවා ගනී
    regressor = pipeline.named_steps["regressor"]
    importances = regressor.feature_importances_
    
    table = (
        pd.DataFrame({"feature": FEATURES, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return table


def evaluate_model(pipeline):
    """පරීක්ෂණ දත්ත (Test set) මත Metrics ගණනය කරයි"""
    df = create_data()
    X = df.drop(columns=["price"])
    y = df["price"]
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = pipeline.predict(X_test)
    
    return {
        "mae_dollars": round(mean_absolute_error(y_test, preds) * 100000, 2),
        "r2_score": round(r2_score(y_test, preds), 4),
    }


def _run_cli():
    """Terminal (CLI) එක හරහා විධාන ක්‍රියාත්මක කිරීමේ කොටස"""
    parser = argparse.ArgumentParser(description="Industrial House Price Predictor CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train and save the pipeline")

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
        pipeline = load_model()
        price = predict_price(
            pipeline,
            medinc=args.medinc, houseage=args.houseage,
            averooms=args.averooms, averageoccupation=args.averageoccupation,
            latitude=args.latitude, longitude=args.longitude,
            population=args.population, avebedrms=args.avebedrms,
        )
        print(json.dumps({"predicted_price_usd": price}, indent=2))
    elif args.command == "importance":
        pipeline = load_model()
        print(feature_importance(pipeline).to_string(index=False))
    elif args.command == "evaluate":
        pipeline = load_model()
        print(json.dumps(evaluate_model(pipeline), indent=2))


if __name__ == "__main__":
    _run_cli()