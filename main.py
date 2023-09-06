from flask import Flask, render_template, url_for, redirect, request, jsonify, session
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
    
    api_data = session.get('api_data')

    if api_data:
      print("Man it totally does!")
      print(api_data)
    
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
          name1 = first_3_entries[0]['Name']
          country1 = first_3_entries[0]['Country']
          currency1 = first_3_entries[0]['Currency']
          code2 = first_3_entries[1]['Code']
          exchange2 = first_3_entries[1]['Exchange']
          name2 = first_3_entries[1]['Name']
          country2 = first_3_entries[1]['Country']
          currency2 = first_3_entries[1]['Currency']
          code3 = first_3_entries[2]['Code']
          exchange3 = first_3_entries[2]['Exchange']
          name3 = first_3_entries[2]['Name']
          country3 = first_3_entries[2]['Country']
          currency3 = first_3_entries[2]['Currency']
          
          codes = [code1, code2, code3]
          
          return render_template('index.html',
                                candle_form=candle_form,
                                pattern_form=pattern_form,
                                ticker_form=ticker_form,
                                codes=codes,
                                exchange1=exchange1,
                                exchange2=exchange2,
                                exchange3=exchange3,
                                name1=name1,
                                name2=name2,
                                name3=name3,
                                country1=country1,
                                country2=country2,
                                country3=country3,
                                currency1=currency1,
                                currency2=currency2,
                                currency3=currency3,
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
    
    
    
    
@app.route('/select_company', methods=['POST'])
def select_company():
    selected_ticker = request.form.get('selected_ticker')
    print(selected_ticker)
    # Handle the selected company and perform API search here
    # Make the API call and process the data as needed
    
    # Store the API data in the session
    session['api_data'] = "I hope this works man"

    # Redirect back to the index route
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)