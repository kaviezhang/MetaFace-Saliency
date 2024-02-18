# MetaFace-Saliency

# Dataset
<p align = "center">    
<img  src="https://github.com/kaviezhang/MetaFace-Saliency/blob/main/Figs/main.png" width="400" />
</p>
We establish an eye-tracking database with 780 synthetic face images and collect fixation data from 22 participants. The synthetics allow for precise merging of expression animations onto the facial skeleton, resulting in more controlled and consistent experimental stimuli. The dataset can be downloaded from [Drive-Google]: https://drive.google.com/drive/folders/1t1Yf0hRAFRyH7HCIDACmnNftQDo09qP_?usp=sharing .


## Model
<p align = "center">    
<img  src="https://github.com/kaviezhang/MetaFace-Saliency/blob/main/Figs/model.png" width="800" />
</p>
We develop a model for predicting saliency maps on synthetic face images. Especially in the testing of both our MetaFace and SALICON datasets, our method achieved state-of-the-art results. This will serve as a reference for attention distribution prediction in photo-realistic facial mesh model in virtual reality applications or games.

## Training & Testing on MetaFace databases
'''
train_metadataset.py

## Training & Testing on SALICON databases
'''
train_salicon.py

