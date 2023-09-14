import tensorflow as tf
import numpy as np
import pandas as pd
from utils.preprocessing.preprocessing import *
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import math
from utils.cdl_patterns.cdl_patterns import CDL_PATTERNS
import ipywidgets as widgets

def get_chart_p(ticker, start_date, end_date,  with_pattern=False, with_candle = False):
    df = yf.download(ticker, start=start_date, end=end_date)
    pretrained_model_p = tf.keras.models.load_model("utils/model_p")
    pretrained_model_d = tf.keras.models.load_model("utils/model_p_d")

    if with_candle == True:
        df = one_candle(df)
        if with_pattern == False:
            return df

    else:
        df = dochl(df)

    if with_pattern == True:
        data = np.array(df.iloc[:,1:5])
        data_preprocessed, scaler = preprocess_X(data)
        ax = popping_pattern(data_preprocessed, pretrained_model_p, pretrained_model_d, scaler)
        for item in ax:
            if item[4] == [0, 0, 0, 0, 0]: item[4] = "No pattern"
            if item[4] == [0, 0, 0, 0, 1]: item[4] ="No pattern"
            if item[4] == [0, 0, 0, 1, 0]: item[4] ="Rising wedge"
            if item[4] == [0, 0, 1, 0, 0]: item[4] ="Falling wedge"
            if item[4] == [0, 1, 0, 0, 0]: item[4] ="Double top"
            if item[4] == [1, 0, 0, 0, 0]: item[4] ="Double bottom"
            if item[2] == [1, 0, 0, 0, 0]: item[2] ="Double bottom"
            item[0] = np.round(item[0]).astype(np.int16)
            item[1] = np.round(item[1]).astype(np.int16)
            if item[4]!= "No pattern":
                high = data[item[0]:item[1]+2,1]
                low = data[item[0]:item[1]+2,2]
                item[2] = round(max(high),1)
                item[3] = round(min(low),1)

        return df, ax

    return df

def plotting(ticker, start_date, end_date,  with_pattern=False, with_candle = False, cdle_patterns=pd.Series(CDL_PATTERNS.keys()).sample(2)):
    if with_pattern == True:

        df, ax = get_chart_p(ticker, start_date, end_date,  with_pattern=with_pattern, with_candle = with_candle)
        data=[]

        for pattern in ax:

            if pattern[4] != 'No pattern':


                start_index = str(df.iloc[pattern[0]]["date"]).replace(" 00:00:00","")
                end_index = str(df.iloc[pattern[1]-1]["date"]).replace(" 00:00:00","")
                lower_bound = pattern[2]
                upper_bound = pattern[3]
                # Create the rectangle coordinates
                x_rect = [start_index, end_index, end_index, start_index, start_index]
                y_rect = [lower_bound, lower_bound, upper_bound, upper_bound, lower_bound]

                # Create a scatter trace for the rectangle
                first = np.random.randint(0,255)
                second = np.random.randint(0,255)
                third = np.random.randint(0,255)
                data.append(go.Scatter(
                    x=x_rect,
                    y=y_rect,
                    mode='lines',
                    fill='toself',  # Fill the area enclosed by the lines
                    fillcolor=f'rgba({first}, {second}, {third}, 0.1)',  # Set the fill color and opacity
                    #line=dict(color='blue'),  # Set the line color
                    name=pattern[4]  # Name for the legend
                ))

        if data==[]:
            print("Couldn't find relevant patterns, try another date range or ticker")
    # Create a scatter trace for some other data (optional)
    # This is just to demonstrate the rectangle within a plot
        data.append(go.Candlestick(x=df.iloc[:,0],
                open=df.iloc[:,1],
                high=df.iloc[:,2],
                low=df.iloc[:,3],
                close=df.iloc[:,4]))


    elif (with_candle == False) & (with_pattern == False):
        df = get_chart_p(ticker, start_date, end_date,  with_pattern=with_pattern, with_candle = with_candle)
        data=[]
        data.append(go.Candlestick(x=df.iloc[:,0],
                open=df.iloc[:,1],
                high=df.iloc[:,2],
                low=df.iloc[:,3],
                close=df.iloc[:,4]))

    if (with_candle == True) & (with_pattern == False):
        df = get_chart_p(ticker, start_date, end_date,  with_pattern=with_pattern, with_candle = with_candle)
        data = []
        data.append(go.Candlestick(x=df.iloc[:,0],
                open=df.iloc[:,1],
                high=df.iloc[:,2],
                low=df.iloc[:,3],
                close=df.iloc[:,4]))

    data = data
    # Create the figure
    fig = go.Figure(data=data)



    if with_candle==True:
        dat = create_data_stock(ticker, start_date, end_date, patterns = cdle_patterns)

        fig = create_candlestick_chart(fig, dat)

    fig.show()




def popping_pattern(data_preprocessed, pretrained_model_p, pretrained_model_d, scaler):

    end_index=[0]
    ends_index=[0]

    lis = tf.convert_to_tensor([data_preprocessed], np.float32)

    a1=1
    while list(data_preprocessed[0][a1]) != [-1, -1, -1, -1]:
        a1+=1
    _,counts = np.unique(lis[0][:,0], return_counts=True)
    count = counts.sum()-counts.max()
    pattern=[]
    ax=[]

    i=0

    while (lis.shape[0]-1) >= i:
        if i == 0:
            data=[]


        sequence = lis[i]

        a=1
        while list(sequence[0][a]) != [-1, -1, -1, -1]:
            a+=1


        if i==0:
            saturation=0
        if a<round(a1*0.1):
            saturation+=1
        if saturation == lis.shape[0]:

            break




        y_p = pretrained_model_p.predict(sequence)
        pattern = list(np.round(y_p)[0].astype(np.int16))
        y_d = pretrained_model_d.predict(sequence)[1]
        start = np.round(y_d[0][0]).astype(np.int16)
        end = np.round(y_d[0][1]).astype(np.int16)

        # if len(end_index) == 0:
        #     end_index.append(end)

        if (pattern == [0, 0, 0, 0, 1]) | (pattern == [0, 0, 0, 0, 0]) | (a<round(a1*0.1)) | (start>end) | (start<0) | (end<0):

            start = round(a/2)
            end = round(a/2)
            ax.append([0, 0, 0, 0, [0, 0, 0, 0, 1]])



        lip = scaler.inverse_transform(sequence[0][:a])

        if (start != end) & (start>0) & (end>0) & (start<end):



            if end>a:
                end = a
            ends = end+end_index[i]
            starts = start + end_index[i]
            if ends > a1:
                ends = a1
            if (starts<ends)&((ends-starts)>5):
                low = int(min(lip[start:end,1]))
                high = int(max(lip[start:end,2]))
                ax.append([starts, ends, low, high, pattern])

        else:
            ends = end + end_index[i]
        position = (i*2)+1
        ends_index.insert(position, ends)
        if len(ax)>1:
            if ax[-1] == ax[-2]:
                ax.pop(-1)

                break




        if (i==0) & ((lis.shape[0]-1) == 0):

            st = sequence[0,:start]
            st = tf.convert_to_tensor([st], np.float32)
            st = pad_sequences(st, maxlen = 450, dtype='float32', padding='post', value=-1)

            en = sequence[0,end:]
            en = tf.convert_to_tensor([en], np.float32)
            en = pad_sequences(en, maxlen = 450, dtype='float32', padding='post', value=-1)


            lis = tf.convert_to_tensor([st, en], np.float32)
            end_index = ends_index.copy()

            i=0
            continue

        if (lis.shape[0]-1) > 0:
            if i == 0:
                data=[]

            if a<round(a1*0.1):
                a = tf.convert_to_tensor([[[-1,-1,-1,-1],[-1,-1,-1,-1]]], np.float32)
                a = pad_sequences(a, maxlen = 450, dtype='float32', padding='post', value=-1)
                a = tf.convert_to_tensor([a, a], np.float32)
                data.append(a)
            else:


                st = sequence[0,:start]
                st = tf.convert_to_tensor([st], np.float32)
                st = pad_sequences(st, maxlen = 450, dtype='float32', padding='post', value=-1)

                en = sequence[0,end:]
                en = tf.convert_to_tensor([en], np.float32)
                en = pad_sequences(en, maxlen = 450, dtype='float32', padding='post', value=-1)


                lid = tf.convert_to_tensor([st, en], np.float32)

                data.append(lid)
            if (i>0) & ((lis.shape[0]-1)>0) & (i == (lis.shape[0]-1)):

                end_index = ends_index.copy()

                lis = tf.concat(data, axis=0)
                i=0
                continue

        if ((lis.shape[0]-1) > 0):

            i=i+1

    return ax

def create_data_stock(ticker, start_date, end_date, patterns = pd.Series(CDL_PATTERNS.keys()).sample(2)):
    dat = yf.download(ticker, start=start_date, end=end_date).reset_index()
    for pattern in patterns:
        calculated_value = CDL_PATTERNS[pattern](dat['Open'], dat['High'], dat['Low'], dat['Close'])
        if calculated_value is not None:
            dat[pattern] = calculated_value

    return dat
def create_candlestick_chart(fig, dat):
    # Filter out patterns that are not present in the data
    arrow_annotation=[]
    for index, row in dat.iterrows():
        for i in range(dat.iloc[:,7:].shape[1]):
            if row[7+i] != 0:
                # arrow_annotation.append(go.layout.Annotation(
                #                         x=row['Date'],
                #                         y=row['High'],
                #                         text=dat.columns[7+i],
                #                         showarrow=True,
                #                         arrowhead=2,
                #                         ax=0,
                #                         ay=-40
                #                     ))
                annotation_dict = {
                'x': row['Date'],
                'y': row['High'],
                'text': dat.columns[7 + i],
                'showarrow': True,
                'arrowhead': 2,
                'ax': 0,
                'ay': -40
                }
                arrow_annotation.append(annotation_dict)
    with fig.batch_update():
            fig.layout.annotations = arrow_annotation
            fig.update_layout(xaxis_rangeslider_visible=False)
                                    # Add the arrow annotation to the layout
    #fig.update_layout(annotations=[arrow_annotation])

    return fig

def one_candle(df):
    df = dochl(df)
    df["stoc1"] = stoc1(df)
    df["O-C"] = min_av(df, 15, entr = "rest", func="mean")
    df["AVGH10"] = min_av(df, 10, entr = "H", func="mean")
    df["AVGL10"] = min_av(df, 10, entr = "L", func="mean")
    df["AVGH10.2"] = offset2(df["AVGH10"])
    df["AVGL10.2"] = offset2(df["AVGL10"])
    df["AVGL20"] = min_av(df, 20, entr = "L", func="mean")
    df["AVGH20"] = min_av(df, 20, entr = "H", func="mean")
    df["MINL10"] = min_av(df, 10, entr = "L", func="min")
    df["MINL5"] = min_av(df, 5, entr = "L", func="min")
    df["h1"] = h1(df)
    df["l1"] = l1(df)
    df["o1"] = o1(df)
    df["c1"] = c1(df)
    df["o2"] = ochl2(df, p="O")
    df["h2"] = ochl2(df, p="H")
    df["l2"] = ochl2(df, p="L")
    df["c2"] = ochl2(df, p="C")
    df["o3"] = o3(df)
    df["c3"] = c3(df)


    df["doji"] = doji(df)
    df["o_marubozu"] = o_marubozu(df)
    df["gravestone"] = gravestone(df)
    df["dragonfly_doji"] = dragonfly_doji(df)
    df["hammer"] = hammer(df)
    df["c_marubozu"] = c_marubozu(df)
    df["bear_engulfing"] = bear_engulfing(df)
    df["bull_engulfing"] = bull_engulfing(df)
    df["tweezer_bottom"] = tweezer_bottom(df)
    df["tweezer_top"] = tweezer_top(df)
    df["morning_star"] = morning_star(df)
    df["evening_star"] = evening_star(df)
    df["three_inside_up"] = three_inside_up(df)
    df["three_inside_down"] = three_inside_down(df)
    df["three_black_crows"] = three_black_crows(df)
    df["three_white_soldiers"] = three_white_soldiers(df)
    return df

def dochl(df):
    df = df.iloc[:,:4]
    df = df.reset_index()
    df.columns = ["date", "open", "high", "low", "close"]
    return df

def ochl(df):
    df = df.iloc[:,:5]
    df.columns = ["date", "open", "high", "low", "close"]
    return df["open"], df["close"], df["high"], df["low"]

def doji(df):
    return (10 * abs(df["open"] - df["close"]) <= df["high"] - df["low"])

def stoc1(df):
    O, C, H, L = ochl(df)
    return ((C-L)/(H-L))*100

def bear_engulfing(df):
    O, C, H, L = ochl(df)
    return (df["o1"] > df["c1"]) & (10 * (C - O) >= 7 * (H - L)) & (C > df["o1"])& (df["c1"] > O) & (10 * (H - L) >= 12 * (df["AVGH10"] - df["AVGL10"]))

def bull_engulfing(df):
    O, C, H, L = ochl(df)
    return (df["o1"] > df["c1"]) & (10 * (C - O) >= 7 * (H - L)) & (C > df["o1"]) & (df["c1"] > O) & (10 * (H - L) >= 12 * (df["AVGH10"] - df["AVGL10"]))

def tweezer_bottom(df):
    O, C, H, L = ochl(df)
    return (L == df["l1"]) & (5 * abs(C - O) < abs(df["c1"] - df["o1"])) & (10 * abs(df["c1"] - df["o1"]) >= 9 * (df["h1"] - df["l1"])) & (10 * (df["h1"] - df["l1"]) >= 13 * (df["AVGH20"] - df["AVGL20"]))

def tweezer_top(df):
    O, C, H, L = ochl(df)
    return (H == df["h1"]) & (abs(C - O) < .2 * abs(df["c1"] - df["o1"])) & (abs(df["c1"] - df["o1"]) >= .9 * (df["h1"] - df["l1"])) & (df["h1"] - df["l1"] >= 1.3 * (df["AVGH20"] - df["AVGL20"]))

def morning_star(df):
    O, C, H, L = ochl(df)
    return (df["o2"] > df["c2"]) & (5 * (df["o2"] - df["c2"]) > 3 * (df["h2"] - df["l2"])) & (df["c2"] > df["o1"]) & (2 * abs(df["o1"] - df["c1"]) < abs(df["o2"] - df["c2"])) & (df["h1"] - df["l1"] > 3 * (df["c1"] - df["o1"])) & (C > O) & (O > df["o1"]) & (O > df["c1"])

def evening_star(df):
    O, C, H, L = ochl(df)
    return (df["c2"] - df["o2"] >= .7 * (df["h2"] - df["l2"])) & (df["h2"] - df["l2"] >= df["AVGH10.2"] - df["AVGL10.2"]) & (df["c1"] > df["c2"]) & (df["o1"] > df["c2"]) & (H - L >= df["AVGH10"] - df["AVGL10"]) & (O - C >= .7 * (H - L)) & (O < df["o1"]) & (O < df["c1"])

def three_inside_up(df):
    O, C, H, L = ochl(df)
    return (10 * (df["o2"] - df["c2"]) >= 7 * (df["h2"] - df["l2"])) & ((df["h2"] - df["l2"]) >= df["AVGH10.2"] - df["AVGL10.2"]) & (df["c1"] > df["o1"]) & (df["o1"] > df["c2"]) & (df["c1"] < df["o2"]) & (5 * (df["c1"] - df["o1"]) <= 3 * (df["o2"] - df["c2"])) & (O > df["o1"]) & (O < df["c1"]) & (C > df["c1"]) & (10 * (C - O) >= 7 * (H - L))


def three_inside_down(df):
    O, C, H, L = ochl(df)
    return (abs(df["c2"] - df["o2"]) > .5 * (df["h1"] - df["l1"])) & (df["c2"] > df["o2"]) & (df["c1"] < df["o1"]) & (df["h1"] < df["c2"]) & (df["l1"] > df["o2"]) & (C < O) & (C < df["c1"])

def three_black_crows(df):
    O, C, H, L = ochl(df)
    return (df["o1"] < df["o2"]) & (df["o1"] > df["c2"]) & (O < df["o1"]) & (O > df["c1"]) & (df["c1"] < df["l2"]) & (C < df["l1"]) & (df["c2"] < 1.05 * df["l2"]) & (df["c1"] < 1.05 * df["l1"]) & (C < 1.05 * L)

def three_white_soldiers(df):
    O, C, H, L = ochl(df)
    return (C > df["c1"]) & (df["c1"] > df["c2"]) & (C > O) & (df["c1"] > df["o1"]) & (df["c2"] > df["o2"]) & (2 * abs(df["c2"] - df["o2"]) > df["h2"] - df["l2"]) & (2 * abs(df["c1"] - df["o1"]) > df["h1"] - df["l1"]) & (H - L > df["AVGH20"] - df["AVGL20"]) & (O > df["o1"]) & (O < df["c1"]) & (df["o1"] > df["o2"]) & (df["o1"] < df["c2"]) & (df["o2"] > df["o3"]) & (df["o2"] < df["c3"]) & (20 * C > 17 * H) & (20 * df["c1"] > 17 * df["h1"]) & (20 * df["c2"] > 17 * df["h2"])

def min_av(df, p, entr = ["C", "H", "L", "rest"], func=["mean","min"]):
    O, C, H, L = ochl(df)
    rest = abs(O-C)
    if entr == "C" : windows = C.rolling(p)
    if entr == "L" : windows = L.rolling(p)
    if entr == "H" : windows = H.rolling(p)
    if entr == "O" : windows = O.rolling(p)
    if entr == "rest" : windows = rest.rolling(p)
    if func == "mean": moving_averages = windows.mean()
    if func == "min": moving_averages = windows.min()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[p - 1:]
    rest = df.shape[0]-len(final_list)



    for i in range(round(rest)):

        final_list = [final_list[0]] + final_list


    return pd.Series(final_list)

def min_avg(df, p, entr = ["C", "H", "L", "rest"], func=["mean","min"]):
    O, C, H, L = ochl(df)
    rest = abs(O-C)
    if entr == "C" : windows = C.rolling(p)
    if entr == "L" : windows = L.rolling(p)
    if entr == "H" : windows = H.rolling(p)
    if entr == "O" : windows = O.rolling(p)
    if entr == "rest" : windows = rest.rolling(p)
    if func == "mean": moving_averages = windows.mean()
    if func == "min": moving_averages = windows.min()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[p - 1:]
    rest = df.shape[0]-len(final_list)

    if rest%2 == 0:

        for i in range(round(rest/2)):
            final_list.append(final_list[-1])
            final_list = [final_list[0]] + final_list
    else:

        for i in range(round(rest/2)+1):
            final_list.append(final_list[-1])
            if i>0:
                final_list = [final_list[0]] + final_list


    return pd.Series(final_list)

def o_marubozu(df):
    O = df["open"]
    C = df["close"]
    H = df["high"]
    L = df["low"]
    return ((L == O) | (O == H)) & (H - L > abs(O - C)) & (abs(O - C) > 3 * df["O-C"] / 2)

def o1(df):
    O, C, H, L = ochl(df)
    df = O[1:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def o3(df):
    O, C, H, L = ochl(df)
    df = O[3:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df[df.shape[0]+1] = df[df.shape[0]]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def c3(df):
    O, C, H, L = ochl(df)
    df = C[3:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df[df.shape[0]+1] = df[df.shape[0]]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def offset2(column_df):
    df = column_df[2:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def ochl2(df, p=["C", "H", "L", "O"]):
    O, C, H, L = ochl(df)
    if p == "C" : df = C[2:]
    if p == "H" : df = H[2:]
    if p == "L" : df = L[2:]
    if p == "O" : df = O[2:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def c1(df):
    O, C, H, L = ochl(df)
    df = C[1:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def h1(df):
    O, C, H, L = ochl(df)
    df = H[1:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def l1(df):
    O, C, H, L = ochl(df)
    df = L[1:]
    df[df.shape[0]+1] = df[df.shape[0]]
    df = df.reset_index(drop=True)
    return df

def gravestone(df):
    O = df["open"]
    C = df["close"]
    H = df["high"]
    L = df["low"]
    return ((100 * abs(O - C)) <= H - L) & (df["stoc1"] <= 5) & (H > L) & (10 * L <= 3 * df["h1"] + 7 * df["l1"]) & (H - L >= df["AVGH10"]-df["AVGL10"])

def dragonfly_doji(df):
    O = df["open"]
    C = df["close"]
    H = df["high"]
    L = df["low"]
    return (50 * abs(O - C) <= H - L) & (df["stoc1"] >= 70) & (H - L >= df["AVGH10"] - df["AVGL10"]) & (L == df["MINL10"])

def hammer(df):
    O, C, H, L = ochl(df)
    return (5 * abs(C - O) <= H - L) & (10 * abs(O - C) >= H - L) & (2 * O >= H + L) & (df["stoc1"] >= 50) & ((20 * O >= 19 * H + L) | (df["stoc1"] >= 95)) & (10 * (H - L) >= 8 * (df["AVGH10"] - df["AVGL10"])) & (L == df["MINL5"]) & (H > L)

def c_marubozu(df):
    O, C, H, L = ochl(df)
    return ((L == C) | (C == H)) & (H - L > abs(O - C)) & (abs(O - C) > 3 * df["O-C"] / 2)
