from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pickle, os, string, nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer

app = Flask(__name__)
app.secret_key = '1c8073775dbc85a92ce20ebd44fd6a4fd832078f59ef16ec'  # secure secret key

ps = PorterStemmer()

# ============================
# ✅ Model and Vectorizer Load
# ============================
tfidf_path = os.path.join(os.getcwd(), 'vectorizer.pkl')
model_path = os.path.join(os.getcwd(), 'model.pkl')

try:
    tfidf = pickle.load(open(tfidf_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    print("❌ Error loading model/vectorizer:", e)
    tfidf = None
    model = None

nltk.download('punkt')

# =====================================================
# 🔄 Auto-retrain if old pickles are incompatible/not fit
# =====================================================
def bootstrap_model_if_needed(force=False):
    global tfidf, model

    needs_bootstrap = (
        force or tfidf is None or model is None or not hasattr(tfidf, "vocabulary_")
    )

    if not needs_bootstrap:
        return

    try:
        csv_path = os.path.join(os.getcwd(), 'spam.csv')
        df = pd.read_csv(csv_path, encoding='latin-1')
        # Standard dataset columns: v1=label, v2=text
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

        # Preprocess text similarly to runtime
        df['clean_text'] = df['text'].fillna('').apply(transform_text)
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_text'], df['label'].map({'ham': 0, 'spam': 1}).astype(int),
            test_size=0.2, random_state=42, stratify=df['label']
        )

        vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2))
        X_train_vec = vectorizer.fit_transform(X_train)

        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)

        # Assign to globals
        tfidf = vectorizer
        model = clf

        # Persist updated artifacts to avoid retrain next boot
        try:
            pickle.dump(tfidf, open(tfidf_path, 'wb'))
            pickle.dump(model, open(model_path, 'wb'))
        except Exception:
            # Non-fatal if writing fails (e.g., permissions)
            pass

        print("✅ Model re-trained from spam.csv and ready.")
    except Exception as e:
        print("❌ Failed to bootstrap model:", e)

bootstrap_model_if_needed()

# ============================================
# ✅ Temporary in-memory user store (no DB use)
# ============================================
users = {}

# ============================
# ✅ Text Preprocessing Helper
# ============================
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# ============================
# ✅ Routes
# ============================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    if 'user' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('signin'))

# ============================
# ✅ Prediction Route (Fixed)
# ============================
@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form.get('message')

    if not input_sms:
        return jsonify({'prediction': "⚠️ Please enter a message."})

    if not tfidf or not model:
        return jsonify({'prediction': "❌ Model or vectorizer not loaded properly."})

    transformed_sms = transform_text(input_sms)

    try:
        if not hasattr(tfidf, "vocabulary_"):
            # Attempt to (re)bootstrap then continue
            bootstrap_model_if_needed(force=True)

        try:
            vector_input = tfidf.transform([transformed_sms])
        except Exception as e:
            # If transform fails due to not-fitted issues, rebuild and retry once
            if "not fitted" in str(e).lower():
                bootstrap_model_if_needed(force=True)
                vector_input = tfidf.transform([transformed_sms])
            else:
                raise

        result = model.predict(vector_input)[0]
        prediction = "🚫 Spam Message" if result == 1 else "✅ Not Spam"

        # return JSON for AJAX
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'prediction': f"Error: {e}"})



# ============================
# ✅ Authentication Routes
# ============================
@app.route('/signin')
def signin():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/register', methods=['POST'])
def register():
    full_name = request.form['full_name']
    username = request.form['username']
    email = request.form['email']
    phone = request.form['phone']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if password != confirm_password:
        flash('Passwords do not match', 'error')
        return redirect('/signup')

    users[email] = {
        'full_name': full_name,
        'username': username,
        'email': email,
        'phone': phone,
        'password': password
    }

    flash('Registration successful! Please log in.', 'success')
    return redirect('/signin')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    remember_me = request.form.get('remember_me')

    user = users.get(email)

    if user and user['password'] == password:
        session['user'] = user
        if remember_me:
            session.permanent = True
        return redirect(url_for('index'))
    else:
        flash('Login failed. Check your email or password.', 'error')
        return redirect('/signin')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

# ============================
# ✅ Run Flask App
# ============================
if __name__ == '__main__':
    app.run(debug=True)
