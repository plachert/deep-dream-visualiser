from __future__ import annotations

import streamlit as st
from activation_tracker.activation import SUPPORTED_FILTERS
from activation_tracker.model import ModelWithActivations
import pathlib

from deepdream.config import SUPPORTED_CONFIGS
from deepdream.image_processing import channel_last
from deepdream.image_processing import convert_to_255scale
from deepdream.image_processing import run_pyramid
from deepdream.image_processing import create_random_image
from deepdream.image_processing import load_image_from


def run():
    with st.spinner('Processing...'):
        images = run_deepdream(
            image_path=image_path,
            config_name=model_selection,
            strategy_name=strategy_selection,
            strategy_params=strategy_params,
            jitter_size=jitter_size,
            octave_n=octave_n,
            octave_scale=octave_scale,
            n_iterations=n_iterations,
            regularization_coeff=regularization_coeff,
            lr=lr,
        )
        st.session_state['images'] = images


@st.cache_data
def get_strategy_params(config_name, strategy_name):
    config = SUPPORTED_CONFIGS[config_name]
    model = config.classifier
    example_input = config.example_input
    activation_filter_class = SUPPORTED_FILTERS[strategy_name]
    model_with_activations = ModelWithActivations(
        model=model, example_input=example_input,
    )
    activations = model_with_activations.activations["all"]
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
        model=classifier, activation_filters={"filtered": [activation_filter]},
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
    images = [deprocessor(image) for image in images]#[convert_to_base64(deprocessor(image)) for image in images]
    return images


if __name__ == '__main__':
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='auto',
        page_title='DeepDream Visualiser',
        page_icon=None,
    )

    st.sidebar.title('DeepDream Visualiser')
    is_disabled = not st.session_state.get('strategy_params', [])
    st.sidebar.button('Run DeepDream', on_click=run, disabled=is_disabled)

    available_model_configs = list(SUPPORTED_CONFIGS.keys())
    available_strategies = list(SUPPORTED_FILTERS.keys())

    model_table, deepdream_table, image_table = st.sidebar.tabs(
        ['Model params', 'DeepDream params', 'Image params'],
    )

    with model_table:
        model_selection = st.selectbox('Select model', available_model_configs)
        strategy_selection = st.selectbox(
            'Select strategy', available_strategies,
        )
        strategy_params = st.multiselect(
            'Select strategy params', get_strategy_params(
                model_selection, strategy_selection,
            ),
            key='strategy_params',
        )

    with deepdream_table:
        jitter_size = st.number_input('Jitter Size', 0, 60, 30, 1)
        octave_n = st.number_input('Pyramid levels', 0, 10, 3, 1)
        octave_scale = st.number_input('Octave scale', 1., 2., 1.4, 1.)
        n_iterations = st.number_input('Iterations per level', 1, 300, 10, 1)
        regularization_coeff = st.number_input(
            'Regularization coeff', 0., 1., 0.1, 0.05,
        )
        lr = st.number_input('Learning Rate', 0.001, 1., 0.01, 0.001)

    with image_table:
        uploaded_file = st.file_uploader(
            'Upload an image', type=['jpg', 'png'],
        )
        if uploaded_file is not None:
            with open(f'examples/uploaded/{uploaded_file.name}', 'wb') as f:
                f.write(uploaded_file.read())
            image_path = f'examples/uploaded/{uploaded_file.name}'
        else:
            image_path = None

    placeholder = st.empty()
    images = st.session_state.get('images')
    if images:
        placeholder.image(
            channel_last(
                convert_to_255scale(images[-1]),
            ), 'Processed Image',
        )
