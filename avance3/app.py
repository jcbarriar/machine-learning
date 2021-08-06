import flask
from numpy import dtype
import pandas as pd
from joblib import dump, load

with open(f'model/modelo_movieratingprediction.joblib','rb') as f:
    model = load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        year = flask.request.form['year']
        age = int(flask.request.form['age'])
        directors = int(flask.request.form['directors'])
        genres = int(flask.request.form['genres'])
        language = int(flask.request.form['language'])
        runtime = flask.request.form['runtime']

    input_variables = pd.DataFrame([[year, age, directors, genres, language, runtime]], columns=['year', 'age', 'directors', 'genres', 'language', 'runtime'], 
                                    dtype='int', 
                                    index=['input'])

    predictions = model.predict(input_variables)[0]
    print(predictions)

    return flask.render_template('main.html', original_input={'Year': year, 'Age': age, 'Directors': directors, 'Genres': genres, 'Language': language, 'Runtime': runtime}, 
                                result=predictions)

if __name__ == '__main__':
    app.run(debug=True)