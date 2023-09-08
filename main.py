from flask import Flask, render_template, url_for, redirect, request, jsonify, session
from flask_bootstrap import Bootstrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly
import ipywidgets as widgets

import requests

import datetime
from dateutil.relativedelta import relativedelta

from utils.one_candle.one_candle import plotting
from utils.candle_patterns.candle_patterns import create_candlestick_chart
from utils.cdl_patterns.cdl_patterns import CDL_PATTERNS

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
      print(api_data)
    
    #---Form Submissions---#  
    if request.method == 'POST':
        #---Candle Form---#  
        if candle_form.validate():
            from_date = candle_form.from_date.data
            to_date = candle_form.to_date.data
            candle_ticker = candle_form.company_ticker.data
            selected_candlesticks = candle_form.selected_candle.data
            print(selected_candlesticks)
            url_base = f"http://34.79.179.30:8000/candlestick/{candle_ticker}/{from_date}/{to_date}"

            res = requests.get(url_base)
            text = res.text
            data = json.loads(text)
            data = data['data']
            df = pd.DataFrame(data)
            
            if not df.empty:
                # Create the candlestick chart and checkboxes
                fig = plotting(ticker=candle_ticker, 
                               start_date=from_date,
                               end_date=to_date,
                               with_pattern=True,
                               with_candle=True,
                               cdle_patterns=selected_candlesticks,
                          )

                # # Customize the layout and display the chart
                # fig.update_layout(
                #     title="Interactive Candlestick Chart with Selected Patterns",
                #     xaxis_title="Date",
                #     yaxis_title="Price",
                #     xaxis_rangeslider_visible=True
                # )

                
                graphJSONCandles = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                return render_template('index.html',
                    candle_form=candle_form,
                    pattern_form=pattern_form,
                    ticker_form=ticker_form,
                    graphJSONCandles=graphJSONCandles
                )
            # ----
            print(df.columns)
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
          
          return render_template('index.html',
                                candle_form=candle_form,
                                pattern_form=pattern_form,
                                ticker_form=ticker_form,
                                code1=code1,
                                code2=code2,
                                code3=code3,
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
    
    # df = pd.DataFrame({
    #   'x': ['1', '2', '3', '4', '5'],
    #   'open': [20, 25, 30, 35, 40],
    #   'low': [15, 20, 25, 30, 35],
    #   'high': [25, 30, 35, 40, 45],
    #   'close': [22, 27, 32, 37, 42]
    # })
    # fig = go.Figure(data=[go.Candlestick(x=df['x'],
    #             open=df['open'],
    #             high=df['high'],
    #             low=df['low'],
    #             close=df['close'])])
    # graphJSONCandles = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html',
                           candle_form=candle_form,
                           pattern_form=pattern_form,
                           ticker_form=ticker_form,
                          #  graphJSONCandles=graphJSONCandles,
                           )
    
    
    
    
@app.route('/select_company', methods=['POST'])
def select_company():
    selected_ticker = request.form.get('selected_ticker')
    print(f"this is what is selected: {selected_ticker}")
    if selected_ticker:
      yesterday_str = datetime.datetime.now() - datetime.timedelta(days=1)
      one_year_ago_str = yesterday_str - relativedelta(years=1)
      yesterday = yesterday_str.strftime("%Y-%m-%d")
      one_year_ago = one_year_ago_str.strftime("%Y-%m-%d")

      res = plotting(ticker=selected_ticker,
                          start_date=one_year_ago,
                          end_date=yesterday,
                          with_pattern=True,
                          )
      graphJSONPAttern = json.dumps(res, cls=plotly.utils.PlotlyJSONEncoder)
      
      return redirect(url_for('index', graphJSONPAttern=graphJSONPAttern))
    

    # Store the API data in the session
    session['api_data'] = "I hope this works man"

    # Redirect back to the index route
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)