from flask import Flask, render_template, request, jsonify, redirect, url_for

from flask_wtf import FlaskForm 
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed
from wtforms import SelectField, SubmitField, IntegerField, FloatField
from deepdream.config import SUPPORTED_CONFIGS
from deepdream.model import SUPPORTED_FILTERS, ModelWithActivations
from functools import lru_cache
from deepdream.image_processing import load_image_from, img2base64, create_random_image, channel_last, create_gif
import pathlib
from werkzeug.utils import secure_filename


@lru_cache
def get_strategy_params(config_name, strategy_name):
    model = SUPPORTED_CONFIGS[config_name].classifier
    activation_filter = SUPPORTED_FILTERS[strategy_name]([])
    model_with_activations = ModelWithActivations(model, activation_filter)
    return model_with_activations.strategy_parameters


available_model_configs = list(SUPPORTED_CONFIGS.keys())
available_strategies = list(SUPPORTED_FILTERS.keys())
image_arrays = [
    channel_last(load_image_from(pathlib.Path("examples/sky.jpg"))),
    channel_last(load_image_from(pathlib.Path("examples/sky.jpg"))),
]

image_viewer_data = {"image_list": image_arrays}
        

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['UPLOAD_FOLDER'] = 'examples/uploaded'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}


class DeepDreamParametersForm(FlaskForm):
    model_choices = [(model_config_name, model_config_name) for model_config_name in available_model_configs]
    strategy_choices = [(strategy_name, strategy_name) for strategy_name in available_strategies]
    model = SelectField('model', choices=model_choices, validators=[DataRequired()]) 
    strategy = SelectField('strategy', choices=strategy_choices, validators=[DataRequired()]) 
    strategy_params = SelectField('strategy_params', choices=[], validators=[DataRequired()])
    image = FileField('Upload Image', validators=[FileAllowed(app.config['ALLOWED_EXTENSIONS'], 'Images only!')])
    jitter_size = IntegerField('Jitter Size', default=30)
    octave_n = IntegerField('Octave N', default=2)
    octave_scale = FloatField('Octave Scale', default=1.4)
    n_iterations = IntegerField('Number of Iterations', default=10)
    run_deepdream = SubmitField('Run DeepDream')
    
    
# New route to render the main page
@app.route('/reset', methods=['GET'])
def reset():
    return redirect(url_for('index'))


@app.route('/', methods=['GET', 'POST'])
def index():
    global image_viewer_data
    deepdream_parameters_form = DeepDreamParametersForm()
    gif_url = ""
    if request.method == 'GET':
        config_name = deepdream_parameters_form.model.choices[0][0]
        strategy_name = deepdream_parameters_form.strategy.choices[0][0]
        deepdream_parameters_form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]

    if request.method == 'POST':
        config_name = deepdream_parameters_form.model.data
        strategy_name = deepdream_parameters_form.strategy.data
        deepdream_parameters_form.strategy_params.choices = [(param, param) for param in get_strategy_params(config_name, strategy_name)]
        image = deepdream_parameters_form.image.data
        file_path = None
        if image is not None:
            filename = secure_filename(image.filename)
            file_path = app.config['UPLOAD_FOLDER'] + '/' + filename
            image.save(file_path)

        if deepdream_parameters_form.run_deepdream.data:
            gif_url = create_gif(image_viewer_data['image_list'], "processed.gif")
            
    return render_template(
        'index.html', 
        deepdream_parameters_form=deepdream_parameters_form, 
        gif_url=gif_url,
        )
    
    
@app.route('/strategy_params/<model>/<strategy>')
def strategy_params(model, strategy):
    params = get_strategy_params(model, strategy)
    return jsonify({'strategy_params' : params})

if __name__ == '__main__':
    app.run(debug=True)