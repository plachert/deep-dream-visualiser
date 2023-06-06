"""Streamlit app for experimenting with DeepDream algorithm."""
from __future__ import annotations

import pathlib

import streamlit as st
from activation_tracker.activation import SUPPORTED_FILTERS
from activation_tracker.model import ModelWithActivations

import deepdream.image_processing as img_proc
from deepdream.config import SUPPORTED_CONFIGS


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
        st.session_state['last_run_params'] = {
            'Image path': image_path,
            'Model': model_selection,
            'Strategy': strategy_selection,
            'Strategy params': strategy_params,
            'Jitter size': jitter_size,
            'Pyramid levels': octave_n,
            'Octave scale': octave_scale,
            'Number of iterations': n_iterations,
            'Regularization coeff': regularization_coeff,
            'Learning rate': lr,
        }


@st.cache_data
def get_strategy_params(config_name, strategy_name):
    config = SUPPORTED_CONFIGS[config_name]
    model = config.classifier
    example_input = config.example_input
    activation_filter_class = SUPPORTED_FILTERS[strategy_name]
    model_with_activations = ModelWithActivations(
        model=model, example_input=example_input,
    )
    activations = model_with_activations.activations['all']
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
        model=classifier, activation_filters={'filtered': [activation_filter]},
    )
    if image_path is None:
        input_image = img_proc.create_random_image()
    else:
        input_image = img_proc.load_image_from(image_path)
    input_image = processor(input_image)
    images = img_proc.run_pyramid(
        model=model_with_activations,
        image=input_image,
        jitter_size=jitter_size,
        octave_n=octave_n,
        octave_scale=octave_scale,
        n_iterations=n_iterations,
        regularization_coeff=regularization_coeff,
        lr=lr,
    )
    images = [
        img_proc.channel_last(img_proc.convert_to_255scale(deprocessor(image)))
        for image in images
    ]
    return images


if __name__ == '__main__':
    uploaded_path = pathlib.Path('examples/uploaded')
    uploaded_path.mkdir(parents=True, exist_ok=True)
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='auto',
        page_title='DeepDream Visualiser',
        page_icon=None,
    )

    st.sidebar.title('Configuration')
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
        jitter_size = st.number_input('Jitter size', 0, 60, 30, 1)
        octave_n = st.number_input('Pyramid levels', 1, 10, 3, 1)
        octave_scale = st.number_input('Octave scale', 1., 2., 1.4, 0.1)
        n_iterations = st.number_input('Iterations per level', 1, 300, 20, 1)
        regularization_coeff = st.number_input(
            'Regularization coeff', 0., 1., 0.1, 0.05,
        )
        lr = st.number_input('Learning rate', 0.001, 1., 0.5, 0.01)

    with image_table:
        uploaded_file = st.file_uploader(
            'Upload an image', type=['jpg', 'png'],
        )
        if uploaded_file is not None:
            with open(f'{uploaded_path}/{uploaded_file.name}', 'wb') as f:
                f.write(uploaded_file.read())
            image_path = f'{uploaded_path}/{uploaded_file.name}'
        else:
            image_path = None

    images = st.session_state.get('images')
    last_run_params = st.session_state.get('last_run_params')

    if images:
        l_margin, image_col, r_margin = st.columns([1, 3, 1])
        n_images = len(images)
        with l_margin:
            st.write('')
        with image_col:
            img_slider = st.slider('Image slider', 1, n_images, n_images)
            st.image(
                images[img_slider-1], 'Processed Image',
                width=600, use_column_width=True,
            )
        with r_margin:
            st.write('')
        with st.expander('Parameters'):
            params_str = {
                param: str(value)
                for param, value in last_run_params.items()
            }
            st.table(params_str)
