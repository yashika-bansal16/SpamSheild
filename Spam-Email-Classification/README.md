# Spam Mail Classification

## Description

The **Spam Mail Classification** project is a web-based application that uses machine learning to classify emails as spam or ham. It features a Flask backend, a frontend created with HTML, CSS, and JavaScript, and a MySQL database for storing user data and email classifications.

### Features

- **Email Classification**: Categorizes incoming emails as spam or ham.
- **User Registration and Login**: Secure account creation and authentication.
- **Real-Time Email Classification**: Classifies emails in real time.
- **User Dashboard**: Users can view their email history and classifications.
- **Machine Learning Model**: Employs a trained model to classify emails.
- **Customization**: Users can configure spam filter settings.

## Technologies Used

- **Flask** (Python Web Framework): For the backend server.
- **HTML, CSS, and JavaScript** (Frontend): For the user interface.
- **MySQL** (Database): For storing user data and email classifications.
- **Machine Learning Libraries** (e.g., Scikit-Learn): Used to build and deploy the email classification model.

## Getting Started

To use the Spam Mail Classification app, follow these steps:

1. **Clone this Repository**: Get the project source code by cloning this repository to your local machine.

2. **Set Up the Flask Backend**:
   - This project runs a Flask server; no external DB is required for basic usage.

3. **Install Required Python Packages**:
   - Recommended: create and activate a virtual environment, then install dependencies.

   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Install deps
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Flask App**:
   - Start the Flask app by running the following command in your terminal:

   ```bash
   python app.py
   ```

   - Open the app in your browser:

   ```bash
   http://127.0.0.1:5000/
   ```

## Open and Run in VS Code

1. Open VS Code and install the “Python” extension from Microsoft.
2. File → Open Folder… → select the `Spam-Email-Classification` folder.
3. Create a virtual environment (if you don’t already have one):
   - Terminal → New Terminal, then run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Select the interpreter:
   - Press Ctrl+Shift+P → “Python: Select Interpreter” → choose the one from `.venv` (or `venv`) inside this project.
5. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
6. Run the app:
   - Option A: Terminal → `python app.py`
   - Option B: Press F5 (Run and Debug), choose “Python File” when prompted to start debugging `app.py`.
7. Open `http://127.0.0.1:5000` in your browser.

Notes:
- The app auto-downloads the NLTK “punkt” tokenizer on first run if missing.
- If you see scikit-learn pickle warnings, the app can retrain a fresh model from `spam.csv` automatically.



