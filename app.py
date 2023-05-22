from flask import Flask, render_template, request, jsonify

from flask_wtf import FlaskForm 
from wtforms import SelectField
from deepdream.config import SUPPORTED_CONFIGS
from deepdream.model import SUPPORTED_FILTERS, ModelWithActivations
from functools import lru_cache

@lru_cache
def get_strategy_params(config_name, strategy_name):
    model = SUPPORTED_CONFIGS[config_name].classifier
    activation_filter = SUPPORTED_FILTERS[strategy_name]([])
    model_with_activations = ModelWithActivations(model, activation_filter)
    return model_with_activations.strategy_parameters


available_model_configs = list(SUPPORTED_CONFIGS.keys())
available_strategies = list(SUPPORTED_FILTERS.keys())

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'


class Form(FlaskForm):
    model_choices = [(model_config_name, model_config_name) for model_config_name in available_model_configs]
    strategy_choices = [(strategy_name, strategy_name) for strategy_name in available_strategies]
    model = SelectField('model', choices=model_choices) 
    strategy = SelectField('strategy', choices=strategy_choices) 
    strategy_params = SelectField('strategy_params', choices=[])

@app.route('/', methods=['GET', 'POST'])
def index():
    form = Form()
    if request.method == 'GET':
        config_name = form.model.choices[0][0]
        strategy_name = form.strategy.choices[0][0]
        form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]

    if request.method == 'POST':
        config_name = form.model.data
        strategy_name = form.strategy.data
        form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]
        form.strategy_params.data = request.form['strategy_params']

    return render_template('index.html', form=form)

@app.route('/strategy_params/<model>/<strategy>')
def strategy_params(model, strategy):
    params = get_strategy_params(model, strategy)
    return jsonify({'strategy_params' : params})

if __name__ == '__main__':
    app.run(debug=True)