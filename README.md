# 🏠 House Price Predictor

AI-powered house price prediction system using Advanced Machine Learning (Gradient Boosting) and a FastAPI backend. This project showcases a production-ready ML pipeline integrated with a modern, responsive Glassmorphism Web UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-brightnessborder)

---

## 🚀 Features
- **Advanced ML Pipeline:** Powered by `Gradient Boosting Regressor` trained on the real-world California Housing dataset.
- **Production-Ready Architecture:** Clean separation of concerns between ML workflows and the API layer.
- **Robust API Backend:** High-performance, asynchronous endpoints powered by FastAPI with strict data validation.
- **Lifespan Management:** Efficient ML model caching during server startup to maximize performance.
- **Futuristic Web UI:** Sleek Single Page Application (SPA) experience featuring a Glassmorphism design and seamless dynamic updates via the JavaScript Fetch API.
- **Dual Interface:** Supports both interactive Web UI execution and Command Line Interface (CLI) automation.

---

## 📂 Project Structure
```text
├── static/
│   └── style.css            # Modern Glassmorphism UI Styles
├── templates/
│   └── index.html           # SPA Web Interface with AJAX
├── model_pipeline.py        # ML Training, Prediction & CLI Management
├── main.py                  # FastAPI Server & Request Lifecycle
└── README.md                # Project Documentation