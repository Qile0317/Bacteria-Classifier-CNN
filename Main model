#cool fat: #%% makes a jupyter block in VScode

import numpy as np
import PIL as Image
from os import *
#function for splitting image into 224 x 224 blocks
def process_image(im):
    imarray = np.array(im)
    im_h, im_w = imarray.shape[:2]
    block_h, block_w = 224, 224
    
    for row in np.arange(im_h - block_h +1, step = block_h):
        for col in np.arange(im_w - block_w +1, step = block_w):
            im1 = imarray[row:row+block_h, col:col+block_w, :]
            im1 = Image.fromarray(im1)
            global i
            global path
            im1.save(path + "\img" + f"{i}" + ".png")
            i+=1
    print("completed")

#sample code for image processing 
#i=0
#path = r"C:\Users\lu_41\Desktop\Bacteria\Staphylococcus.saprophiticus\edited"
#for file in os.listdir(r"C:\Users\lu_41\Desktop\Bacteria\Staphylococcus.saprophiticus\raw"):
    #filename = r"C:\Users\lu_41\Desktop\Bacteria\Staphylococcus.saprophiticus\raw" + f"\{file}"
    #im = Image.open(filename)
    #process_image(im)

#I did NOT remove blank slides as there were way too many folders to be manually looked through. Perhaps I can write a function for automating this?

from fastai import *

from fastai.tabular import *
from fastai.vision import *
import fastai.vision as vision


path = r"C:\Users\lu_41\Desktop\Bacteria\data"

np.random.seed(42)
data = vision.ImageDataBunch.from_folder(path, valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4, bs=32).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

learn = cnn_learner(data, models.resnet50, metrics=accuracy).to_fp32()

learn.fit_one_cycle(4)

learn.save('Bacterial Genus prediction - 4 epoch')
learn.export('Bacterial Genus prediction - 4 epoch')

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(10, max_lr=slice(7e-5, 9e-4))

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,10), dpi=100)

interp.plot_top_losses(6, figsize=(15,15))
