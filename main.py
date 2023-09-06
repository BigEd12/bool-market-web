from flask import Flask, render_template, url_for, redirect, request
from flask_bootstrap import Bootstrap

from wtforms import Form, StringField
from wtforms.validators import InputRequired

from forms import MyForm, PatternForm


#---------------------- FLASK ----------------------#
app = Flask(__name__)
app.config["SECRET_KEY"] = "8BYkEfBA6O6donzWlSihBXox7C0sKR6b"
Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    candle_form = MyForm(request.form)
    pattern_form = PatternForm(request.form)
    candlestick_data = None

    if request.method == 'POST':
        if candle_form.validate():
            from_date = candle_form.from_date.data
            to_date = candle_form.to_date.data
            candle_ticker = candle_form.company_ticker.data

            # candlestick_data = fetch_candlestick_data(candle_ticker, from_date, to_date)
            # candlestick_data = process_candlestick_data(candlestick_data)

    initial_candlestick_data = {
        'x': [1, 2, 3, 4, 5],
        'open': [100, 100, 100, 120, 150],
        'high': [110, 110, 110, 130, 160],
        'low': [90, 90, 90, 90, 110],
        'close': [105, 105, 105, 125, 140],
    }


    return render_template('index.html',
                           candle_form=candle_form,
                           pattern_form=pattern_form,
                           )


if __name__ == "__main__":
    app.run(debug=True)