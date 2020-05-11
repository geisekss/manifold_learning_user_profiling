# Manifold Learning for user profiling based on motion sensor data


A novel user-agnostic approach for identity verification based on motion traits acquired by mobile sensors. The proposed approach does not require user-specific training before deployment in mobile devices nor does it require any extra sensor in the device. This solution is capable of learning a user profiling manifold from a small user subset and extend it to unknown users. We validated the proposal on two public datasets. The reported experiments demonstrate remarkable results under a cross-dataset protocol and an open-set setup. This repository is part of the published paper "Manifold learning for user profiling and identity verification using motion sensors" (PII S0031-3203(20)30211-9).


### Requirements
* Python 3.5
* numpy (>= 1.14)
* scikit-image (0.15.0)
* scipy (1.1.0)
* tensorflow-gpu (1.9.0)
* keras (2.2.2)
* scikit-learn (0.21.2)
* matplotlib (3.0.3)


### Modules
* Create motion-level frames: ```python3 processing_data.py path_dataset subset path_frames train_cnn```.
* Generate images from the motion-level frames: ```python3 processing_data.py path_frames feature path_images```.
* Train CNN models: ```python3 train_cnn_models.py trainDir valDir path_models```.
* Train a meta-classifier to perform late fusion of the models: ```python3 train_meta_classifier.py trainDir valDir path_models```.
* Validate the CNN and meta-classifier models: ```python3 validate_cnn_models.py galleryDir testDir path_models```.


### Datasets, motion-level frames and generated images
 Santos, Geise; Pisani, Paulo Henrique; Leyva, Roberto; Li, Chang-Tsun; Tavares, Tiago; Rocha, Anderson (2020), “Manifold learning for user profiling and identity verification using motion sensors: gait datasets”, Mendeley Data, v3. http://dx.doi.org/10.17632/fwhn8hmz4f.3

### CNN architecture details and models
Santos, Geise; Pisani, Paulo Henrique; Leyva, Roberto; Li, Chang-Tsun; Tavares, Tiago; Rocha, Anderson (2020), “Manifold learning for user profiling and identity verification using motion sensors: CNN-designed architecture and models”, Mendeley Data, v3. http://dx.doi.org/10.17632/mgcgv9ztyb.3 

### Parameters
* `path_dataset`: absolute path of the target dataset (RecodGait v2 or IDNet's). The dataset folder should contain `recod` or `idnet` in the name.
* `subset`: subset of the target dataset to generate the frames `user_coordinates`, `raw_data`, or `subset_train_cnn`.
* `path_frames`: absolute path of a folder where the frames are stored. It is stored the folders `gallery` and `probe` inside this folder, or/and the folders `train_cnn` and `validate_cnn`.
* `train_cnn`: flag to indicate wheter the generated frames are from a subset to train the CNNs. If this is equal to 1, the frames will be stored on the folders `train_cnn` and `validate_cnn`.
* `feature`: there were proposed two different images to input at the CNN model: Spectrogram and Accelerometer Dense Trajectories (ADT) images. Thus, this parameter must be `spectrogram_multiD` or `adt_features_multiD`.
* `path_images`: absolute path of a folder where the images are stored. The images are generated for each of the folds `train_cnn`, `validate_cnn`, `gallery` and `probe`. A folder for each `feature` is created, and also a folder for each user.
* `trainDir`: absolute path of the folder where the images to train the CNN (`train_cnn` fold) are stored. 
* `valDir`: absolute path of the folder where the images to validate the CNN (`validate_cnn` fold) are stored.
* `galleryDir`: absolute path of the folder where the gallery images (`gallery` fold) are stored.
* `testDir`: absolute path of the folder where the proble images (`probe` fold) are stored.
* `path_models`: absolute path of the folder where the CNN models, and meta-classifier model is stored.
