- i may need to retrain with slightly simpler resnets to prevent overfitting bc it looks like its slightly overfitting
- i also should train a net to recognize data that is too empty and create a new flder with images that arent mostly white.
- i also should retrain with different yellow/blue tints to represent different camera lighting/image qualities

- ALSO: maybe make it so that the imput image is repeatedly tested different times with different blue yellow tints and see if they all have similar results. However if sometimes tinting makes it look too much like the other gram color then nvm or tone it down.

- it might help to train several nets with different resolution images and have a simple algo to choose which redily made model to deploy based on img size
- ofc, to improve data, i could try find high quality data of new species but thats hard

# important:
I should inspect my current images to see resolution and quality. So that I can manipulate training data accordingly. It may be low quality and result in me having to alter training data to match resolution and zoom.
