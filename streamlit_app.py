from __future__ import annotations

import streamlit as st

from app import get_strategy_params
from deepdream.config import SUPPORTED_CONFIGS
from deepdream.model import SUPPORTED_FILTERS

st.sidebar.title('Deep Dream Visualiser')

available_model_configs = list(SUPPORTED_CONFIGS.keys())
available_strategies = list(SUPPORTED_FILTERS.keys())


model_table, deepdream_table, image_table = st.sidebar.tabs(
    ['Model params', 'DeepDream params', 'Image Params'],
)

with model_table:
    model_selection = st.selectbox('Select model', available_model_configs)
    strategy_selection = st.selectbox('Select strategy', available_strategies)
    strategy_params = st.multiselect(
        'Select strategy params', get_strategy_params(
            model_selection, strategy_selection,
        ),
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

# from PIL import Image
# import numpy as np
# import time

# dog = Image.open("examples/dog.jpg")
# sky = Image.open("examples/sky.jpg")

# placeholder = st.empty()

# images = 10*[dog, sky]

# while True:
#     for i, image in enumerate(images):
#         placeholder.image(image, f"iter {i}")
#         time.sleep(1)
