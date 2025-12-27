import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import requests
from flask import render_template, flash, request

from config import Config
from webapp import app, forms


@app.route("/", methods=["GET", "POST"])
def home():
    form = forms.MainForm()
    if not app.config["TESTING"]:
        form_validation = form.validate_on_submit()
    else:
        form_validation = request.method == 'POST'

    res = None
    if form_validation:
        try:
            start_phrase = form.text_area_field.data
            selected_model = form.models_list_field.data
            text_len = form.text_len_field.data
            temperature = form.temperature_field.data

            payload = {
                "model_name": selected_model,
                "start_phrase": start_phrase,
                "generation_length": text_len,
                "temperature": temperature
                }
            req = requests.post(f"http://{Config.API_HOST}:{Config.API_PORT}/gen_text", json=payload)
            res = req.json()["res_text"]
        except Exception as e:
            flash(f"Error: {e}", "danger")

    
    return render_template("home.html", form=form, res=res)