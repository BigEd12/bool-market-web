from wtforms import Form, StringField, DateField, RadioField, SelectMultipleField
from wtforms.validators import InputRequired
from utils.cdl_patterns.cdl_patterns import CDL_PATTERNS

choices = []

for key, value in CDL_PATTERNS.items():
    choices.append(key)

class CandleForm(Form):
    from_date = DateField('From (YYYY-MM-DD)', validators=[InputRequired()])
    to_date = DateField('To (YYYY-MM-DD)', validators=[InputRequired()])
    company_ticker = StringField('Company Ticker', validators=[InputRequired()])
    selected_candle = SelectMultipleField('Select Candle Pattern', choices=choices)

class PatternForm(Form):
    company_ticker = StringField('Company Ticker')
    
class TickerForm(Form):
    company_name = StringField('Company name')