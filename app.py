from flask import Flask, render_template, request, jsonify
from recomen_utils import recommend, get_suggestions  # ✅ use your utility functions

app = Flask(__name__)

# ---------------- Home Page ----------------
@app.route('/')
def home():
    return render_template('index.html')

# ---------------- Recommendation Route ----------------
@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_name = request.form['movie']
    corrected_name, recommended, original_input = recommend(movie_name)

    if not recommended:
        return render_template('recommend.html', movie=movie_name, recommendations=[], original_input=original_input)

    return render_template('recommend.html', movie=corrected_name, recommendations=recommended, original_input=original_input)

# ---------------- Suggestion Route for Autocomplete ----------------
@app.route('/suggest', methods=['GET'])  # ✅ fix method to GET
def suggest():
    query = request.args.get('q', '').lower()
    suggestions = get_suggestions(query)
    return jsonify(suggestions)

# ---------------- Start the Server ----------------
if __name__ == '__main__':
    app.run(debug=True)
