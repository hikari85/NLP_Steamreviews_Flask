from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the vectorizer and model
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('mlp_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    review = ''
    top_words = []
    word_count = 0
    confidence = None

    if request.method == 'POST':
        review = request.form['review']
        if review.strip():
            vector = vectorizer.transform([review])
            result = model.predict(vector)[0]
            prediction = "✅ Recommended" if result == 1 else "❌ Not Recommended"
            
            # Confidence score
            proba = model.predict_proba(vector)[0]
            confidence = round(max(proba) * 100, 2)

            # Top influential words
            feature_names = vectorizer.get_feature_names_out()
            top_indices = vector[0].toarray()[0].argsort()[-5:][::-1]
            top_words = [(feature_names[i], round(vector[0, i], 4)) for i in top_indices if vector[0, i] > 0]

            # Word count
            word_count = len(review.split())
        else:
            prediction = "⚠️ Please enter a review."

    return render_template('index.html', prediction=prediction, review=review,
                           top_words=top_words, word_count=word_count,
                           confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
