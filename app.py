from __future__ import annotations

import pathlib
from functools import lru_cache

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from flask_wtf.file import FileField
from werkzeug.utils import secure_filename
from wtforms import FloatField
from wtforms import IntegerField
from wtforms import SelectField
from wtforms import SelectMultipleField
from wtforms import SubmitField
from wtforms.validators import DataRequired

from deepdream.config import SUPPORTED_CONFIGS
from deepdream.image_processing import create_random_image
from deepdream.image_processing import load_image_from
from deepdream.image_processing import run_pyramid
from deepdream.model import ModelWithActivations
from deepdream.model import SUPPORTED_FILTERS

images = []


@lru_cache
def get_strategy_params(config_name, strategy_name):
    config = SUPPORTED_CONFIGS[config_name]
    model = config.classifier
    example_input = config.example_input
    activation_filter_class = SUPPORTED_FILTERS[strategy_name]
    model_with_activations = ModelWithActivations(
        model=model, example_input=example_input,
    )
    activations = model_with_activations.activations
    parameters = activation_filter_class.list_all_available_parameters(
        activations,
    )
    return parameters


def run_deepdream(
    image_path: pathlib.Path | None,
    config_name: str,
    strategy_name: str,
    strategy_params: list,
    jitter_size: int,
    octave_n: int,
    octave_scale: float,
    n_iterations: int,
    lr: float,
    regularization_coeff: float,
):
    activation_filter = SUPPORTED_FILTERS[strategy_name](strategy_params)
    config = SUPPORTED_CONFIGS[config_name]
    classifier = config.classifier
    processor = config.processor
    deprocessor = config.deprocessor
    model_with_activations = ModelWithActivations(
        model=classifier, activation_filter=activation_filter,
    )
    if image_path is None:
        input_image = create_random_image()
    else:
        input_image = load_image_from(image_path)
    input_image = processor(input_image)
    images = run_pyramid(
        model=model_with_activations,
        image=input_image,
        jitter_size=jitter_size,
        octave_n=octave_n,
        octave_scale=octave_scale,
        n_iterations=n_iterations,
        regularization_coeff=regularization_coeff,
        lr=lr,
    )
    # [convert_to_base64(deprocessor(image)) for image in images]
    images = [deprocessor(image) for image in images]
    return images


available_model_configs = list(SUPPORTED_CONFIGS.keys())
available_strategies = list(SUPPORTED_FILTERS.keys())

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['UPLOAD_FOLDER'] = 'examples/uploaded'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}


class DeepDreamParametersForm(FlaskForm):
    model_choices = [
        (model_config_name, model_config_name)
        for model_config_name in available_model_configs
    ]
    strategy_choices = [
        (strategy_name, strategy_name)
        for strategy_name in available_strategies
    ]
    model = SelectField(
        'model', choices=model_choices,
        validators=[DataRequired()],
    )
    strategy = SelectField(
        'strategy', choices=strategy_choices, validators=[DataRequired()],
    )
    strategy_params = SelectMultipleField(
        'strategy_params', choices=[], validators=[DataRequired()],
    )
    uploaded_image = FileField(
        'Upload Image', validators=[
            FileAllowed(
                app.config['ALLOWED_EXTENSIONS'], 'Images only!',
            ),
        ],
    )
    jitter_size = IntegerField('Jitter Size', default=30)
    octave_n = IntegerField('Octave N', default=2)
    octave_scale = FloatField('Octave Scale', default=1.4)
    n_iterations = IntegerField('Number of Iterations', default=10)
    regularization_coeff = FloatField('Regularization coeff', default=0.1)
    lr = FloatField('Learning Rate', default=0.1)
    run_deepdream = SubmitField('Run DeepDream')


@app.route('/', methods=['GET', 'POST'])
def index():
    global images
    deepdream_parameters_form = DeepDreamParametersForm()
    if request.method == 'GET':
        config_name = deepdream_parameters_form.model.choices[0][0]
        strategy_name = deepdream_parameters_form.strategy.choices[0][0]
        deepdream_parameters_form.strategy_params.choices = [
            (param, param) for param in get_strategy_params(config_name, strategy_name)
        ]

    if request.method == 'POST':
        config_name = deepdream_parameters_form.model.data
        strategy_name = deepdream_parameters_form.strategy.data
        deepdream_parameters_form.strategy_params.choices = [
            (param, param) for param in get_strategy_params(config_name, strategy_name)
        ]
        image = deepdream_parameters_form.uploaded_image.data
        file_path = None
        if image is not None:
            filename = secure_filename(image.filename)
            file_path = pathlib.Path(
                app.config['UPLOAD_FOLDER'] + '/' + filename,
            )
            image.save(file_path)
        if deepdream_parameters_form.run_deepdream.data:
            strategy_params = deepdream_parameters_form.strategy_params.data
            jitter_size = deepdream_parameters_form.jitter_size.data
            octave_n = deepdream_parameters_form.octave_n.data
            octave_scale = deepdream_parameters_form.octave_scale.data
            n_iterations = deepdream_parameters_form.n_iterations.data
            regularization_coeff = deepdream_parameters_form.regularization_coeff.data
            lr = deepdream_parameters_form.lr.data
            images = run_deepdream(
                image_path=file_path,
                config_name=config_name,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                jitter_size=jitter_size,
                octave_n=octave_n,
                octave_scale=octave_scale,
                n_iterations=n_iterations,
                regularization_coeff=regularization_coeff,
                lr=lr,
            )

    return render_template(
        'index.html',
        deepdream_parameters_form=deepdream_parameters_form,
        images=images,
    )


@app.route('/strategy_params/<model>/<strategy>')
def strategy_params(model, strategy):
    params = get_strategy_params(model, strategy)
    return jsonify({'strategy_params': params})


if __name__ == '__main__':
    app.run(debug=True)
