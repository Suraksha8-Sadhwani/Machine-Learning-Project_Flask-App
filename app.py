from flask import Flask, render_template, request
from predict import *

app = Flask(__name__)

model = init_model()


@app.route('/')
def render():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        inp = request.form.get('input_data')
        news_status = "This News Seems To Be " + \
            predict_news_status(model, inp)
        return render_template('index.html', status=news_status)


if __name__ == '__main__':
    app.run()
