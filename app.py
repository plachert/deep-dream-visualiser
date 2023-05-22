from flask import Flask, render_template, request, jsonify

from flask_wtf import FlaskForm 
from wtforms import SelectField, SubmitField
from deepdream.config import SUPPORTED_CONFIGS
from deepdream.model import SUPPORTED_FILTERS, ModelWithActivations
from functools import lru_cache
import numpy as np
from deepdream.image_processing import load_image_from, img2base64, create_random_image, channel_last
import pathlib


@lru_cache
def get_strategy_params(config_name, strategy_name):
    model = SUPPORTED_CONFIGS[config_name].classifier
    activation_filter = SUPPORTED_FILTERS[strategy_name]([])
    model_with_activations = ModelWithActivations(model, activation_filter)
    return model_with_activations.strategy_parameters


available_model_configs = list(SUPPORTED_CONFIGS.keys())
available_strategies = list(SUPPORTED_FILTERS.keys())
image_arrays = [
    channel_last(create_random_image()),#load_image_from(pathlib.Path("examples/amazon.jpg"))),
    channel_last(load_image_from(pathlib.Path("examples/sky.jpg"))),
]
current_index = 0


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'


class DeepDreamParametersForm(FlaskForm):
    model_choices = [(model_config_name, model_config_name) for model_config_name in available_model_configs]
    strategy_choices = [(strategy_name, strategy_name) for strategy_name in available_strategies]
    model = SelectField('model', choices=model_choices) 
    strategy = SelectField('strategy', choices=strategy_choices) 
    strategy_params = SelectField('strategy_params', choices=[])


@app.route('/', methods=['GET', 'POST'])
def index():
    global image_arrays, current_index
    deepdream_parameters_form = DeepDreamParametersForm()
    if request.method == 'GET':
        config_name = deepdream_parameters_form.model.choices[0][0]
        strategy_name = deepdream_parameters_form.strategy.choices[0][0]
        deepdream_parameters_form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]

    current_array = image_arrays[current_index]
    current_image = img2base64(current_array)

    if request.method == 'POST':
        if deepdream_parameters_form.validate_on_submit():
            config_name = deepdream_parameters_form.model.data
            strategy_name = deepdream_parameters_form.strategy.data
            deepdream_parameters_form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]
            deepdream_parameters_form.strategy_params.data = request.form['strategy_params']
        elif request.form.get("next") == "NEXT":
            current_index += 1
            print("next", current_index)
            current_array = image_arrays[current_index]
            current_image = img2base64(current_array)
        elif request.form.get("previous") == "PREVIOUS":
            current_index -= 1
            current_array = image_arrays[current_index]
            current_image = img2base64(current_array)
    total_images = len(image_arrays)

    return render_template(
        'index.html', 
        deepdream_parameters_form=deepdream_parameters_form, 
        current_image=current_image,
        )
    
@app.route('/next')
def next_image():
    global image_arrays
    current_index = int(request.args.get('index', 0))
    next_index = current_index + 1

    if next_index >= len(image_arrays):
        next_index = len(image_arrays) - 1

    next_array = image_arrays[next_index]
    next_image = img2base64(next_array)

    total_images = len(image_arrays)
    is_last_index = next_index == total_images - 1

    return render_template(
        'index.html', 
        form=None,
        current_image=next_image, 
        current_index=next_index, 
        total_images=total_images, 
        is_last_index=is_last_index,
        )

@app.route('/previous')
def previous_image():
    global image_arrays
    current_index = int(request.args.get('index', 0))
    previous_index = current_index - 1

    if previous_index < 0:
        previous_index = 0

    previous_array = image_arrays[previous_index]
    previous_image = img2base64(previous_array)

    total_images = len(image_arrays)
    is_last_index = previous_index == total_images - 1

    return render_template(
        'index.html', 
        form=None,
        current_image=previous_image, 
        current_index=previous_index, 
        total_images=total_images, 
        is_last_index=is_last_index,
        )



@app.route('/strategy_params/<model>/<strategy>')
def strategy_params(model, strategy):
    params = get_strategy_params(model, strategy)
    return jsonify({'strategy_params' : params})

if __name__ == '__main__':
    app.run(debug=True)