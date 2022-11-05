# Bacteria Classifier
This is development code for a convolutional neural network that can take in a 224x244 image imput of gram stained bacteria viewed at 100x light magnification and classify the genus and sometimes the strain of the bacteria. There are 30 strains it was trained with, all of which come from a cropped version of the DiBas dataset: https://doi.org/10.5281/zenodo.7293846

# Performance
The model is 98.91% accurate.

<img src="https://github.com/Qile0317/Bacteria-Classifier-CNN/blob/main/ConfusionMatrix.png" width="30%"/>

# Deployment
The model is deployed as a web application at https://huggingface.co/spaces/qile0317/Bacteria-Classification

All app development code is at: https://huggingface.co/spaces/qile0317/Bacteria-Classification/tree/main
