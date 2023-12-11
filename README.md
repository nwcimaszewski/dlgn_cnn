# CNN Identification of dLGN Response to Natural Movies

Github repository for a master's thesis project in the lab of Philipp Berens.

Python libraries developed by the Sinz Lab are fit to minimize Poisson loss between predicted and observed binned spike counts computed from dorsal lateral geniculate nucleus (dLGN) in mouse using multi-electrode arrays.  This data was recorded under the presence of randomized optogenetic intervention suppressing the activity of Layer VI primary visual cortex neurons (and thus silencing feedback activity propagating from V1 to dLGN)  Concurrent recordings of pupil location, pupil dilation, and running speed are also available as model inputs.  Performance metric takes the form of Pearson's correlation between the activity and activation of a final layer's units.

## Installation instructions

This code was developed and experiments were run on the Berenslab CIN machines.  See internal groupwiki for details on creating Docker image and building container.  Once a container is running, all notebooks should be easily runnable.

To install `dlgn_cnn` as a package, use `pip install dlgn_cnn`.

## Dataset creation
This package includes scripts for converting video files and response data collected into pickled pandas dataframes into Deeplake datasets.  These datasets can be stored and managed locally or hosted on openloop.ai.  This allows for jagged tensors (differences in length of e.g. temporal dimension) and built in Pytorch dataloaders to be constructed.  Deeplake datasets organize data into Tensors, which in our case will usually consist of `videos`, `responses`, `behavior`, `pupil_location` (and possibly `optogenetic_signal`).  Deeplake stores data in a format optimized for training deep learning models, and is able to handle jagged tensors (i.e. data with variable length over one or more dimensions).  Deeplake was utilized primarily for compatibility with the Sensorium 2023 competition.

## Dataloader creation

To allow for model fitting to multiple recording sessions/animals, models are trained using a nested dictionary of multiple dataloaders.  These dictionaries have first-level keys of `train`, `val` and `test`, and subsequent secondary-level keys indicating the source of dataset.  These dataloaders are constructed using `deeplake[enterprise]` methods which integrate Pytorch methods, and allow for flexible preprocessing of datasamples betweeen different Tensors. 


## Model creation
This repository primarily uses the `neuralpredictors` library.  This allows for a modular architecture of deep neural networks, comprised of a core module which computes complex features of stimulus using convolution (variants include 2D convolution, rotation equivariant 2D convolution, 3D convolution, and factorized 3D convolution), gated recurrent unit modules which allow for transient temporal information to accrue over subsequent frames of a video, a readout module which computes a spatially localized linear combination of the core output channels to best fit the activity of the target recorded neurons, and a shifter module which adjusts the spatial position of the readout vector depending on pupil fixation location within the visual field.


## Model fitting

To keep track of configurations of dataloaders, models, and trainers, the package `nnfabrik` is employed, which uses Datajoint to store unique constructor/config pairs in hashed tables, initialize these with random seeds, and automatically populate tables with trained model state dicts depending on linked entries of an experiments table.  This package also supports checkpoint of intermediate state dicts throughout the training process, but support for this in `dlgn_cnn` is yet to be added.


## Fitted model analysis

Current project goals are to produce and inspect maximally exciting images on models fit to conditions where corticogeniculate feedback is either suppressed or intact.  Towards this end the `MEI` package published by the Sinz Lab will be integrated into future commits.  Besides this, Trainer objects supporting plotting of predictions over ground truth responses for arbitrary batches of videos, as well as tracking of loss and performance metrics via Weights and Biases.


