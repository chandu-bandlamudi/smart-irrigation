# Smart Irrigation System — Flask Web Version

This is a web conversion of the original Tkinter Smart Irrigation project.

## Features
- CSV dataset upload
- Dataset preprocessing and StandardScaler normalization
- 80/20 train-test split
- Agent–Environment training/evaluation
- Rewards and penalty graph
- Irrigation-status prediction from a CSV

## Local run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Render
Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn app:app`

Do not deploy the `.bat` file; it is only for the original Windows desktop version.
