from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Environment import Environment
from Agent import Agent
import io
import base64
import os


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

state = {
    "dataset": None,
    "X": None, "Y": None,
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "scaler": None,
    "env": None, "agent": None,
    "rewards": None, "penalty": None
}

def reset_training_state():
    state.update({
        "X": None, "Y": None, "X_train": None, "X_test": None,
        "y_train": None, "y_test": None, "scaler": None,
        "env": None, "agent": None, "rewards": None, "penalty": None
    })

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        return jsonify(ok=False, message="Please upload a CSV dataset."), 400
    try:
        df = pd.read_csv(file)
        df.fillna(0, inplace=True)
        if "class" not in df.columns:
            return jsonify(ok=False, message="Dataset must contain a 'class' column."), 400
        state["dataset"] = df
        reset_training_state()
        labels = df["class"].astype(str).value_counts().to_dict()
        return jsonify(
            ok=True,
            message="Smart Irrigation Dataset loaded successfully.",
            filename=file.filename,
            rows=len(df),
            columns=list(df.columns),
            preview=df.head(10).to_html(classes="table", index=False),
            labels=labels
        )
    except Exception as e:
        return jsonify(ok=False, message=f"Could not read dataset: {e}"), 400

@app.route("/preprocess", methods=["POST"])
def preprocess():
    df = state["dataset"]
    if df is None:
        return jsonify(ok=False, message="Upload a dataset first."), 400
    try:
        values = df.values
        if values.shape[1] < 8:
            return jsonify(ok=False, message="The supplied project expects at least 8 dataset columns."), 400
        X = values[:, 1:7]
        Y = values[:, 7]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        state["X"], state["Y"], state["scaler"] = X, Y, scaler
        state.update({"X_train": None, "X_test": None, "y_train": None, "y_test": None,
                      "env": None, "agent": None, "rewards": None, "penalty": None})
        return jsonify(ok=True, message="Dataset Processing & Normalization Completed.",
                       samples=int(X.shape[0]), features=int(X.shape[1]))
    except Exception as e:
        return jsonify(ok=False, message=f"Preprocessing failed: {e}"), 400

@app.route("/split", methods=["POST"])
def split():
    if state["X"] is None:
        return jsonify(ok=False, message="Preprocess the dataset first."), 400
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            state["X"], state["Y"], test_size=0.2, random_state=42
        )
        state.update(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        return jsonify(ok=True, message="Dataset Train & Test Split completed.",
                       train_samples=len(X_train), test_samples=len(X_test))
    except Exception as e:
        return jsonify(ok=False, message=f"Split failed: {e}"), 400

@app.route("/train", methods=["POST"])
def train():
    if state["X_train"] is None:
        return jsonify(ok=False, message="Split the dataset before training."), 400
    try:
        env = Environment()
        agent = Agent(env)
        rewards, penalty = agent.step(
            state["X_train"], state["y_train"],
            state["X_test"], state["y_test"]
        )
        state.update(env=env, agent=agent, rewards=int(rewards), penalty=int(penalty))
        return jsonify(ok=True, message="Reinforcement Learning Completed.",
                       rewards=int(rewards), penalty=int(penalty),
                       accuracy=round(100 * rewards / max(len(state["X_test"]), 1), 2))
    except Exception as e:
        return jsonify(ok=False, message=f"Training failed: {e}"), 400

@app.route("/graph")
def graph():
    if state["rewards"] is None:
        return jsonify(ok=False, message="Train the model first."), 400

    try:
        rewards = int(state["rewards"])
        penalty = int(state["penalty"])

        max_value = max(rewards, penalty, 1)

        reward_width = int((rewards / max_value) * 500)
        penalty_width = int((penalty / max_value) * 500)

        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg"
             width="650" height="350"
             viewBox="0 0 650 350">

            <rect width="650" height="350" fill="white"/>

            <text x="325" y="40"
                  text-anchor="middle"
                  font-size="24"
                  font-family="Arial"
                  font-weight="bold">
                Rewards &amp; Penalty
            </text>

            <text x="60" y="105"
                  font-size="18"
                  font-family="Arial">
                Rewards
            </text>

            <rect x="130" y="80"
                  width="{reward_width}"
                  height="40"
                  rx="6"
                  fill="#2e8b57"/>

            <text x="{140 + reward_width}" y="107"
                  font-size="16"
                  font-family="Arial">
                {rewards}
            </text>

            <text x="60" y="190"
                  font-size="18"
                  font-family="Arial">
                Penalty
            </text>

            <rect x="130" y="165"
                  width="{penalty_width}"
                  height="40"
                  rx="6"
                  fill="#d9534f"/>

            <text x="{140 + penalty_width}" y="192"
                  font-size="16"
                  font-family="Arial">
                {penalty}
            </text>

            <line x1="60" y1="250"
                  x2="590" y2="250"
                  stroke="#333"/>

            <text x="325" y="295"
                  text-anchor="middle"
                  font-size="16"
                  font-family="Arial">
                Smart Irrigation Model Evaluation
            </text>
        </svg>
        """

        return svg, 200, {"Content-Type": "image/svg+xml"}

    except Exception as e:
        return jsonify(
            ok=False,
            message=f"Graph generation failed: {str(e)}"
        ), 400

@app.route("/predict", methods=["POST"])
def predict():
    if state["agent"] is None:
        return jsonify(ok=False, message="Train the model before prediction."), 400
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        return jsonify(ok=False, message="Please upload a CSV prediction file."), 400
    try:
        data = pd.read_csv(file)
        data.fillna(0, inplace=True)
        temp = data.values
        test_data = temp[:, 1:7]
        test_data = state["scaler"].transform(test_data)
        predictions = []
        for i in range(len(test_data)):
            pred = state["agent"].predictCondition(
                state["X_train"], state["y_train"], test_data[i]
            )
            predictions.append({
                "row": i + 1,
                "data": [str(x) for x in temp[i]],
                "status": str(pred)
            })
        return jsonify(ok=True, message="Irrigation prediction completed.",
                       predictions=predictions)
    except Exception as e:
        return jsonify(ok=False, message=f"Prediction failed: {e}"), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
