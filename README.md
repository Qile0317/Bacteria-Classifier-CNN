# Bacteria Classifier
This is development code for a convolutional neural network that can take in a 224x244 image imput of gram stained bacteria viewed at 100x light magnification and classify the genus and sometimes the strain of the bacteria. There are 30 strains it was trained with, all of which come from the DiBas dataset: https://github.com/gallardorafael/DIBaS-Dataset

(Note that the cropped dataset is not perfect with some relatively blank images, and removal of those has high potential to further improve performance.)

# Performance
The model is 98.98% accurate.

# Deployment - In progress
The model is deployed as a web application at https://huggingface.co/spaces/qile0317/Bacteria-Classification/tree/main
