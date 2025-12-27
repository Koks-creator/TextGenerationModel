import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField, SelectField, IntegerField, FloatField
from wtforms.validators import DataRequired, ValidationError, length, NumberRange
from flask_wtf.file import MultipleFileField, FileAllowed

from config import Config
from webapp import available_models


class MainForm(FlaskForm):
    text_area_field = TextAreaField("Paste yout starting phrase",
                                [length(max=1000)],
                                render_kw={"rows": 15},
                                default="Hello"
                                )
    models_list_field = SelectField("Styles list", choices=[(m, m) for m in available_models])
    text_len_field = IntegerField("Length of generated text", 
                                  validators=[NumberRange(min=Config.MODEL_MIN_GEN_LEN,
                                                          max=Config.MODEL_MAX_GEN_LEN)],
                                  render_kw={"min": Config.MODEL_MIN_GEN_LEN,
                                             "max": Config.MODEL_MAX_GEN_LEN,
                                             "step": 10},
                                  default=200
                                  )
    temperature_field = FloatField("Temperature",
                                validators=[NumberRange(min=Config.MODEL_MIN_TEMPERATURE,
                                                        max=Config.MODEL_MAX_TEMPERATURE)],
                                render_kw={"min": Config.MODEL_MIN_TEMPERATURE,
                                           "max": Config.MODEL_MAX_TEMPERATURE,
                                           "step": 0.1},
                                default=0.7
                                )
    submit_field = SubmitField("Submit")