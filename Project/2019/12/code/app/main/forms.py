from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, PasswordField, SelectField, TextAreaField, HiddenField,validators
from wtforms.validators import DataRequired, Length, Email, Regexp, EqualTo


class SearchForm(FlaskForm):
    check_order = StringField('排序', validators=[DataRequired(message='不能为空'), Length(0, 64, message='长度不正确')])
    sentimentReview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

    submit = SubmitField('查询')

class SummaryForm(FlaskForm):
    texttitle = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=1)])
    textcontent = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=10)])

    submit = SubmitField('查询')

