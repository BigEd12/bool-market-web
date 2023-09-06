from wtforms import Form, StringField, DateField
from wtforms.validators import InputRequired

class MyForm(Form):
    from_date = DateField('From (YYYY-MM-DD)', validators=[InputRequired()])
    to_date = DateField('To (YYYY-MM-DD)', validators=[InputRequired()])
    company_ticker = StringField('Company Ticker', validators=[InputRequired()])


class PatternForm(Form):
    company_ticker = StringField('Company Ticker', validators=[InputRequired()])