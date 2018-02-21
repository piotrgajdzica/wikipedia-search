import json
from flask import Flask, render_template, request, make_response
from search import get_matrix, get_base_words, get_svd_results, get_results

app = Flask(__name__)


def get_articles(question):
    return question


@app.route('/nosvd', methods=['GET', 'POST'])
def nosvd():
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        return json.dumps(get_results(datafromjs, A, base_words))


@app.route('/file', methods=['GET', 'POST'])
def get_file():
    if request.method == 'POST':
        datafromjs = request.form['filename']

        response = make_response(str(open("./data/articles/" + datafromjs + ".txt", encoding='UTF-8').read()))
        response.headers["content-type"] = "text/plain"

        return response


@app.route('/svd', methods=['GET', 'POST'])
def svd():
    print("heelo")
    if request.method == 'POST':
        datafromjs = request.form['mydata']

        k = int(datafromjs.split()[0])
        question = " ".join(datafromjs.split()[1:])

        return json.dumps(get_svd_results(datafromjs, k))


@app.route("/")
def main():
    return render_template('index.html')

A = get_matrix()
base_words = get_base_words()

if __name__ == "__main__":
    app.run(debug=True)
