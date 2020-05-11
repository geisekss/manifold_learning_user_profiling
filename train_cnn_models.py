from keras import *
from keras.preprocessing.image import *
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.callbacks import *
from keras import backend as K
import os, pickle, sys, json, glob, math
from settings import * 
from random import *
from collections import Counter
import tensorflow as tf
from sklearn import metrics


def tp_rate(y_true, y_pred):
    total_p = K.sum(K.cast(K.equal(y_true[:, 1], 1), 'int32'))
    tp = K.equal(y_true[:, 1], 1) & K.equal(K.round(y_pred[:, 1]), 1)
    tp = K.sum(K.cast(K.reshape(tp, [-1]), 'int32'))
    return tp/total_p


def tn_rate(y_true, y_pred):
    total_n = K.sum(K.cast(K.equal(y_true[:, 0], 1), 'int32'))
    tn = K.equal(y_true[:, 0], 1) & K.equal(K.round(y_pred[:, 0]), 1)
    tn = K.sum(K.cast(K.reshape(tn, [-1]), 'int32'))
    return tn/total_n


def acc_norm(y_true, y_pred):
    tpr = tp_rate(y_true, y_pred)
    tnr = tn_rate(y_true, y_pred)
    return (tpr + tnr)/2


def getl_label_imgName(imgName):
    return int(imgName.split("/")[-2])


def get_ID_imgName(imgName):
    return imgName.split("_")[-1].split('.')[0]


def sampling_uniform(labelsImages, user, n_samples_neg):
        total_classes = np.unique(labelsImages)
        idxs_class_neg_orig = (labelsImages != int(user)).nonzero()[0]
        neg_classes = np.delete(total_classes, np.where(total_classes==int(user))[0])
        n_samples_class = int(math.floor(n_samples_neg/float(len(neg_classes))))
        seed(1)
        idxs_class_neg = np.hstack(([np.array(sample(list(np.where(labelsImages==c)[0]), n_samples_class), dtype=int) if(len(np.where(labelsImages==c)[0]) > n_samples_class) else list(np.where(labelsImages==c)[0]) for c in neg_classes]))
        idxs_class_neg = np.hstack((idxs_class_neg, np.array(sample(list(set(idxs_class_neg_orig)-set(idxs_class_neg)), n_samples_neg-len(idxs_class_neg)), dtype=int) ))
        return idxs_class_neg

def get_pairs(image_paths_train, image_paths_val, labels_images_val):
    pairs = []
    for img in image_paths_train:
        label_img = getl_label_imgName(img)
        idxs_pos_val = np.where(np.array(labels_images_val)==label_img)[0]
        pairs_pos = [[image_paths_val[idx], img, 1] for idx in idxs_pos_val]
        seed(0)
        idxs_neg_val = np.where(np.array(labels_images_val)!=label_img)[0].tolist()
        if len(idxs_neg_val) > len(idxs_pos_val):
            idxs_neg_val = sample(idxs_neg_val, len(idxs_pos_val))
        pairs_neg = [[image_paths_val[idx], img, 0] for idx in idxs_neg_val]
        pairs += [val for pair in zip(pairs_pos, pairs_neg) for val in pair]
    return pairs
        

def get_pairs_intra_session(image_paths, labels_images):
    pairs = []
    for idxImg in range(len(image_paths)):
        idxs_pos = np.where(np.array(labels_images)==labels_images[idxImg])[0]
        pairs_pos = [[image_paths[idx], image_paths[idxImg], 1] for idx in idxs_pos if idx != idxImg]
        seed(0)
        idxs_neg = np.where(np.array(labels_images)!=labels_images[idxImg])[0].tolist()
        if len(idxs_neg) > len(idxs_pos):
            idxs_neg = sample(idxs_neg, len(idxs_pos))
        pairs_neg = [[image_paths[idx], image_paths[idxImg], 0] for idx in idxs_neg]
        pairs += [val for pair in zip(pairs_pos, pairs_neg) for val in pair]
    return pairs


def load_dataset_batches(split_dir_train, split_dir_val, nbClasses, batch_size=128, users_valid=None):
    #Get all file names for that split
    if(users_valid):
        users = [u for u in os.listdir(split_dir_train) if len(u.split('.'))==1]
        filenames_train = [img for user in users if user in users_valid for img in glob.glob(split_dir_train+user+"/*") if img[-3:] in ['jpg', 'png', 'jpeg']]
        filenames_val = [img for user in users if user in users_valid for img in glob.glob(split_dir_val+user+"/*") if img[-3:] in ['jpg', 'png', 'jpeg']]
    else:
        filenames_train = [img for img in glob.glob(split_dir_train+"*/*") if img[-3:] in ['jpg', 'png', 'jpeg']]
        filenames_val = [img for img in glob.glob(split_dir_val+"*/*") if img[-3:] in ['jpg', 'png', 'jpeg']]

    nInBatch = 0
    image_paths_train = sample(filenames_train, len(filenames_train))
    image_paths_val = sample(filenames_val, len(filenames_val))
    labels_images_train = [getl_label_imgName(img) for img in image_paths_train]
    labels_images_val = [getl_label_imgName(img) for img in image_paths_val]
    pairs_inter = get_pairs(image_paths_train, image_paths_val, labels_images_val)
    pairs_intra_train = get_pairs_intra_session(image_paths_train, labels_images_train)
    pairs_intra_val = get_pairs_intra_session(image_paths_val, labels_images_val)
    pairs = pairs_inter + pairs_intra_train + pairs_intra_val

    while True:
        batch, labels = [], []        
        pairs = sample(pairs, len(pairs))
        pairs_pos = [pair for pair in pairs if pair[2]==1]
        pairs_neg = [pair for pair in pairs if pair[2]==0]
        pairs = [val for pair in zip(pairs_pos, pairs_neg) for val in pair]

        for i in range(len(pairs)):
                img1 = img_to_array(load_img(pairs[i][0], target_size=(224, 224)))
                img2 = img_to_array(load_img(pairs[i][1], target_size=(224, 224)))
                img = np.concatenate((img1, img2), axis=2)
                label = np_utils.to_categorical(pairs[i][2], nbClasses)               
                batch.append(img)
                labels.append(label)
                nInBatch += 1             
                #if we already have one batch, yields it
                if nInBatch >= batch_size:
                    yield np.array(batch), np.array(labels)
                    batch, labels = [], []
                    nInBatch = 0
        #yield the remaining of the batch
        if nInBatch > 0:
            yield np.array(batch), np.array(labels)


def average_fusion(labels, predictions, ids):
    new_labels, new_predictions = [], []
    for idImage in np.unique(ids):
        idxs_id = np.where(ids==idImage)[0]
        label = list(labels[idxs_id[0]])
        pred = np.mean(predictions[idxs_id], axis=0).tolist()
        pred = pred/np.sum(pred)
        new_labels.append(label)
        new_predictions.append(pred)
    return new_labels, new_predictions


def max_prob_temporal(labels, predictions):
        new_labels, new_predictions, new_ids = [], [], []
        new_predictions = [np.max(predictions[i:i+n_fusion, :], axis=0).tolist() for i in range(len(predictions)-n_fusion)] 
        new_labels = [[Counter(labels[i:i+n_fusion, 0]).most_common()[0][0], Counter(labels[i:i+n_fusion, 1]).most_common()[0][0]] for i in range(len(labels)-n_fusion)]
        return new_labels, new_predictions


def create_cnn_models(nbClasses=2, alpha=0.5):
	shape = (1, 1, int(1024 * alpha))
	base_model1 = applications.mobilenet.MobileNet(weights=None, include_top=False, input_shape=(224,224,6), classes=nbClasses, alpha=alpha)
	base_model2 = applications.mobilenet.MobileNet(weights=None, include_top=False, input_shape=(224,224,6), classes=nbClasses, alpha=alpha)
	x1 = base_model1.output 
	x1 = GlobalAveragePooling2D()(x1)
	x1 = Reshape(shape, name='reshape_1')(x1)
	x1 = Dropout(0.25, name='dropout')(x1)
	x1 = Conv2D(nbClasses, (1, 1), padding='same', name='conv_preds')(x1)
	x1 = Activation('softmax', name='act_softmax')(x1)
	x2 = base_model2.output 
	x2 = GlobalAveragePooling2D()(x2)
	x2 = Reshape(shape, name='reshape_1')(x2)
	x2 = Dropout(0.25, name='dropout')(x2)
	x2 = Conv2D(nbClasses, (1, 1), padding='same', name='conv_preds')(x2)
	x2 = Activation('softmax', name='act_softmax')(x2)
	output1 = Reshape((nbClasses,), name='reshape_2')(x1)
	output2 = Reshape((nbClasses,), name='reshape_2')(x2)
	model1 = Model(inputs=base_model1.input, outputs=output1)
	model2 = Model(inputs=base_model2.input, outputs=output2)
	return model1, model2


def train_model(model, datasetDir_train_cnn, datasetDir_val_cnn, path_models, name_model, nbClasses=2, batch_size=128, steps_epoch=500, val_steps=1000):
	model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy', tp_rate, tn_rate, acc_norm])
	users = [u for u in os.listdir(datasetDir_train_cnn) if len(u.split('.'))==1]
	history_final = {}
	for i in range(20):
	    users_train = sample(users, int(len(users)*0.8))
	    users_val = [u for u in users if u not in users_train]
	    train_cnn_generator = load_dataset_batches(datasetDir_train_cnn, datasetDir_val_cnn, nbClasses, batch_size, users_train)
	    validation_cnn_generator = load_dataset_batches(datasetDir_train_cnn, datasetDir_val_cnn, nbClasses, batch_size, users_val)
	    checkpointer = ModelCheckpoint(path_models+"/"+name_model+"."+str(i+1)+"-{val_loss:.4f}-{acc:.4f}.h5", monitor='val_loss', verbose=0, save_best_only=False, period=1)
	    history = model.fit_generator(train_cnn_generator, steps_per_epoch=steps_epoch, validation_data=validation_cnn_generator, validation_steps=val_steps, epochs=1, callbacks=[checkpointer]) 
	    history_final[i] = history.history
	with open(path_models+"/history_"+name_model+".json", "w") as outfile: 
		json.dump(history_final, outfile, indent=1)



if __name__ == "__main__":
    """
    It trains a CNN model for Spectrogram images and another for the ADT images. This reads these images stored in a `trainDir` and `valDir` to train and validate the CNN models.

    Parameters
    ----------
        trainDir: `string` absolute path of the folder where the images to train the CNN (`train_cnn` fold) are stored. 
        valDir: `string` absolute path of the folder where the images to validate the CNN (`validate_cnn` fold) are stored.
        path_models: `string` absolute path of the folder where the CNN models, and meta-classifier model is stored.
    Output
    -------
        It stores .h5 models in the `path_models` during the training of the two CNNs: Spectrogram and ADT. Also, it stores the training and validation scoring history.
        
    """

	trainDir = sys.argv[1]
	valDir = sys.argv[2]
	path_models = sys.argv[3] 

	model1, model2 = create_cnn_models()
	train_model(model1, trainDir+"/spectrogram_multiD/", valDir+"/spectrogram_multiD/", path_models, "model_spectrogram")
	train_model(model2, trainDir+"/adt_features_multiD/", valDir+"/adt_features_multiD/", path_models, "model_adt")




