from flask import Flask, render_template, url_for, redirect, request, jsonify
from flask_bootstrap import Bootstrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly

import requests

from wtforms import Form, StringField
from wtforms.validators import InputRequired

from forms import CandleForm, PatternForm, TickerForm


#---------------------- FLASK ----------------------#
app = Flask(__name__)
app.config["SECRET_KEY"] = "8BYkEfBA6O6donzWlSihBXox7C0sKR6b"
Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    #---Prepare Forms---#  
    candle_form = CandleForm(request.form)
    pattern_form = PatternForm(request.form)
    ticker_form = TickerForm(request.form)
    candlestick_data = None
    
    
    

    #---Form Submissions---#  
    if request.method == 'POST':
        #---Candle Form---#  
        if candle_form.validate():
            from_date = candle_form.from_date.data
            to_date = candle_form.to_date.data
            candle_ticker = candle_form.company_ticker.data

        #---Ticker Form---#      
        if ticker_form.validate():
          comp_name = ticker_form.company_name.data
          api_url = f'https://eodhistoricaldata.com/api/search/{comp_name}?api_token=64ef583e9a7137.71172021'
          response = requests.get(api_url)
          res_text = response.text
          data = json.loads(res_text)
          first_3_entries = data[:3]

          code1 = first_3_entries[0]['Code']
          exchange1 = first_3_entries[0]['Exchange']
          code2 = first_3_entries[1]['Code']
          exchange2 = first_3_entries[1]['Exchange']
          code3 = first_3_entries[2]['Code']
          exchange3 = first_3_entries[2]['Exchange']
          
          return render_template('index.html',
                                candle_form=candle_form,
                                pattern_form=pattern_form,
                                ticker_form=ticker_form,
                                code1=code1,
                                code2=code3,
                                code3=code3,
                                exchange1=exchange1,
                                exchange2=exchange2,
                                exchange3=exchange3,
                                )
    
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
                           ticker_form=ticker_form,
                           graphJSON=graphJSON,
                           )



if __name__ == "__main__":
    app.run(debug=True)