# CNN Identification of dLGN Response to Natural Movies

Github repository for a master's thesis project in the lab of Philipp Berens.

Python libraries developed by the Sinz Lab are fit to maximize correlation between the activity (currently quantified as binned firing rate, to be expanded to waveform convolved voltage estimates) and activation of a final layer's units.  Much of the experiment running code is .

## Installation instructions

## Dataset creation
This package includes scripts for converting video files and response data collected into pickled pandas dataframes into Deeplake datasets.  These datasets can be stored and managed locally or hosted on openloop.ai.  This allows for jagged tensors (differences in length of e.g. temporal dimension) and built in Pytorch dataloaders to be constructed.  Deeplake datasets organize data into Tensors, which in our case will usually consist of `videos`, `responses`, `behavior`, `pupil_location` (and possibly `optogenetic_signal`).

## Dataloader creation

To allow for model fitting to multiple recording sessions/animals, models are trained using a nested dictionary of multiple dataloaders.  These dictionaries have first-level keys of `train`, `val` and `test`, and subsequent secondary-level keys indicating the source of dataset.  These dataloaders are constructed using `deeplake[enterprise]` methods which integrate Pytorch methods, and allow for flexible transforming of datasampes betweeen different Tensors. 


## Model creation
This repository primarily uses the `neuralpredictors` library.  This allows for a modular architecture of deep neural networks, comprised of a core module which computes complex features of stimulus using convolution, recurrent modules which allow for temporal information to affect the output of the core, a readout module which computes a spatially localized linear combination of the core outputs to best fit the activity of the target recorded neurons, and a shifter module which adjusts the spatial position of the readout vector depending on pupil fixation location within the visual field. 

Each module is defined using a dictionary of configuration hyperparameters, and the entire model is created by passing in these config dicts andthe dataloaders dictionary.  Most of the code for was forked from the sensorium 2023 repository.

## Model fitting

Trainer functions operate on model and dataloader dict objects.  Current approaches use learning rate scheduler which admits a patience (number of epochs without improvement upon previous optimum before decreasing learning rate) and tolerance (number of learning rate decrements before termination).

## Fitted model analysis

Readout vectors can be visualized.  Eventually MEIs will be produced to fitted models.


