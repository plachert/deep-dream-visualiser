from flask import Flask, render_template, request, jsonify

from flask_wtf import FlaskForm 
from wtforms.validators import DataRequired
from wtforms import SelectField, SubmitField
from deepdream.config import SUPPORTED_CONFIGS
from deepdream.model import SUPPORTED_FILTERS, ModelWithActivations
from functools import lru_cache
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
    channel_last(create_random_image()),
    channel_last(load_image_from(pathlib.Path("examples/sky.jpg"))),
]

image_viewer_data = {"image_list": image_arrays, "current_image": img2base64(image_arrays[0]), "current_index": 0}


def update_current_image():
    global image_viewer_data
    current_idx = image_viewer_data["current_index"]
    image_viewer_data["current_image"] = img2base64(image_viewer_data["image_list"][current_idx])

def next_image():
    global image_viewer_data
    current_idx = image_viewer_data["current_index"]
    image_list_size = len(image_viewer_data["image_list"])
    if current_idx + 1 < image_list_size:
        image_viewer_data["current_index"] += 1
        update_current_image()

def previous_image():
    global image_viewer_data
    current_idx = image_viewer_data["current_index"]
    if current_idx - 1 >= 0:
        image_viewer_data["current_index"] -= 1
        update_current_image()
        

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'


class DeepDreamParametersForm(FlaskForm):
    model_choices = [(model_config_name, model_config_name) for model_config_name in available_model_configs]
    strategy_choices = [(strategy_name, strategy_name) for strategy_name in available_strategies]
    model = SelectField('model', choices=model_choices, validators=[DataRequired()]) 
    strategy = SelectField('strategy', choices=strategy_choices, validators=[DataRequired()]) 
    strategy_params = SelectField('strategy_params', choices=[], validators=[DataRequired()])
    run_deepdream = SubmitField('Run DeepDream')
    
    
class ImageViewerForm(FlaskForm):
    previous = SubmitField('Previous')
    next = SubmitField('Next')


@app.route('/', methods=['GET', 'POST'])
def index():
    global image_viewer_data
    deepdream_parameters_form = DeepDreamParametersForm()
    image_viewer_form = ImageViewerForm()
    if request.method == 'GET':
        config_name = deepdream_parameters_form.model.choices[0][0]
        strategy_name = deepdream_parameters_form.strategy.choices[0][0]
        deepdream_parameters_form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]

    if request.method == 'POST':
        config_name = deepdream_parameters_form.model.data
        strategy_name = deepdream_parameters_form.strategy.data
        deepdream_parameters_form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]
        

    return render_template(
        'index.html', 
        deepdream_parameters_form=deepdream_parameters_form, 
        image_viewer_form=image_viewer_form,
        current_image=image_viewer_data["current_image"],
        )
    
    
@app.route('/image_viewer', methods=['POST'])
def image_viewer():
    deepdream_parameters_form = DeepDreamParametersForm()
    image_viewer_form = ImageViewerForm()
    if image_viewer_form.validate_on_submit():
        if image_viewer_form.previous.data:
            previous_image()
        elif image_viewer_form.next.data:
            next_image()
        ...  # handle the login form
    # render the same template to pass the error message
    # or pass `form.errors` with `flash()` or `session` then redirect to /
    return render_template('index.html', deepdream_parameters_form=deepdream_parameters_form, image_viewer_form=image_viewer_form, current_image=image_viewer_data["current_image"])
    


@app.route('/strategy_params/<model>/<strategy>')
def strategy_params(model, strategy):
    params = get_strategy_params(model, strategy)
    return jsonify({'strategy_params' : params})

if __name__ == '__main__':
    app.run(debug=True)