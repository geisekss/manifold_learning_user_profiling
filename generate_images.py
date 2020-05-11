from processing_data import *
from settings import *
from scipy.ndimage import zoom
import random, os, pickle, sys, json, glob, imageio
from skimage import exposure
from scipy.misc import bytescale
from matplotlib import mlab


def spectrogram_multiD(frames_set, **kwargs):
	n = int(frames_set.shape[1]/2.0)
	features = np.array([np.dstack(([mlab.specgram(x=sequence[:, k], NFFT=128, Fs=Fs, noverlap=127, scale_by_freq=True, sides='twosided')[0][:, :-1].T for k in range(frames_set.shape[2])])) for sequence in frames_set])
	return features

def adt_features_multiD(frames_set, L=128,  **kwargs):
    features = np.array([[trajectories_multiD(frames_set[i, k:k+L, :]) for k in range(0, frames_set.shape[1]-L)] for i in range(frames_set.shape[0])])
    return features

def calculate_features(frames_set, meta, feature_function):
	feature_function = eval(feature)
	features = feature_function(frames_set)
	idxs_valid = [i for i in range(len(features)) if len(np.equal(features[i], None).nonzero()[0]) == 0]
	features = np.array([features[idx] for idx in idxs_valid])
	if(features.shape[0] > frames_set.shape[0]):
		p = int(features.shape[0]/frames_set.shape[0])
		meta = np.vstack(([np.repeat([m], p, axis=0) for m in meta]))
	print('features:', features.shape)
	return features, meta[idxs_valid, :]


def prepare_images(images_set, desired_size=(224, 224, 3)):
	initial_shape = images_set.shape[1:]
	ratio = np.array(desired_size, dtype=float)/np.array(initial_shape, dtype=float)
	print(ratio)
	new_images = np.array([zoom(image, tuple(ratio)) for image in images_set])
	return new_images


def read_mlevel_frames(files):
    assert len(files) == 3
    frames = np.dstack(([np.loadtxt(f, delimiter=',')[:, dim_meta:] for f in files]))
    meta = np.loadtxt(files[0], delimiter=',', usecols=(0,1,2,)).astype(int)
    return frames, meta


def save_images(images, meta, path):
    users = np.unique(meta[:, label_user])
    for user in users:
        idxs_user = np.where(meta[:, label_user]==user)[0]
        path_user = path+str(user)+"/"
        if(not os.path.isdir(path_user)):
            os.mkdir(path_user)
        _ = [imageio.imwrite(path_user+str(user)+"_"+str(idx)+".png", bytescale(exposure.rescale_intensity(images[idx], in_range=(0,1)))) for idx in idxs_user]
    


if __name__ == "__main__":
    """
    It generates Spectrogram or ADT images for a given set of motion-level frames. This reads the frames stored as CSV in a folder, in which must contain a CSV file for each acceleration coordinate (x, y and z).

    Parameters
    ----------
        path_frames: `string` absolute path of the target dataset (RecodGait v2 or IDNet's). The dataset folder should contain recod or idnet in the name.
        feature: `string` there were proposed two different images to input at the CNN model: Spectrogram and Accelerometer Dense Trajectories (ADT) images. Thus, this parameter must be `spectrogram_multiD` or `adt_features_multiD`.
        path_images: `string` absolute path of a folder where the images are stored. The images are generated for each of the folds `train_cnn`, `validate_cnn`, `gallery` and `probe`. A folder for each `feature` is created, and also a folder for each user.
    Output
    -------
        It stores `feature` images in the folder `path_images`.

    """

    path_frames = sys.argv[1]
    feature = sys.argv[3]
    path_images = sys.argv[2]+"/"+feature+"/"
    files = glob.glob(path_frames+"*")
    print('Reading the motion-level frames...')
    frames, meta = read_mlevel_frames(files)
    print('Generating images...')
    features, meta = calculate_features(frames, meta, feature)
    print('Saving the images...')
    images = prepare_images(features) 
    if(not os.path.isdir(path_images)):
        os.mkdir(path_images)

    save_images(images, meta, path_images)

