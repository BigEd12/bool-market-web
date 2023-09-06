from flask import Flask, render_template, url_for, redirect, request, jsonify
from flask_bootstrap import Bootstrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly

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
    
    df = pd.DataFrame({
      'x': ['1', '2', '3', '4', '5'],
      'open': [20, 25, 30, 35, 40],
      'low': [15, 20, 25, 30, 35],
      'high': [25, 30, 35, 40, 45],
      'close': [22, 27, 32, 37, 42]
    })
    fig = go.Figure(data=[go.Candlestick(x=df['x'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           candle_form=candle_form,
                           pattern_form=pattern_form,
                           graphJSON=graphJSON
                           )



if __name__ == "__main__":
    app.run(debug=True)