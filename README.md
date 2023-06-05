# DeepDream Visualiser

##  Description


## Getting Started

### Activation Tracker


## Demo
### Visualise learnt features
When applied to a random image, the algorithm  provides insights into the learned features of the model. In the following example we select `TargetsActivationFilter` as a strategy and `71` as a parameter. This way the optimization algorithm will try to maximize the 71st neuron of the last layer which is associated with scorpion class. We can run the algorithm with default parameters.

![](https://github.com/plachert/deep-dream-experiments/blob/streamlit/examples/show_scorpion.gif)

When the image is processed we can see the results. We can examine the transformation process by playing with the slider. As you can see the features that the model found to be useful for recognizing a scorpion in the image make sense. The model seems to have captured the specific features of a scorpion - body segments and twisted legs.

### Amplify features in an input image
We can also run the algorithm on a given input image in order to amplify features. In the following example we amplify all the ReLU activations in the model.

![](https://github.com/plachert/deep-dream-experiments/blob/streamlit/examples/show_sky.gif)

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/plachert/activation_tracker/blob/main/LICENSE)
