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
from sklearn import metrics, model_selection, svm


def getl_label_imgName(imgName):
    return int(imgName.split("/")[-2])

def get_ID_imgName(imgName):
    return imgName.split("_")[-1].split('.')[0]


def generate_batches(pairs, batch_size, nbClasses=2):                                      
    batch, labelList = [], []
    nInBatch = 0
    for i in range(len(pairs)):
        img1 = img_to_array(load_img(pairs[i][0], target_size=(224, 224)))
        img2 = img_to_array(load_img(pairs[i][1], target_size=(224, 224)))
        img = np.concatenate((img1, img2), axis=2)
        label = np_utils.to_categorical(pairs[i][2], nbClasses)
        batch.append(img)
        labelList.append(label)
        nInBatch += 1
        if nInBatch >= batch_size:
            yield np.array(batch), np.array(labelList)
            batch, labelList = [], []
            nInBatch = 0
    if nInBatch > 0:
        yield np.array(batch), np.array(labelList)


def meta_predictions(labels, predictions1, predictions2, ids_imgs, user_imgs):
	new_labels, new_meta, new_user_imgs, new_ids_imgs = [], [], [], []
	for user in np.unique(user_imgs):
		idxs_user = np.where(user_imgs==user)[0]
		ids_user = ids_imgs[idxs_user]
		labels_user = labels[idxs_user, 1]
		predictions1_user = predictions1[idxs_user]
		predictions2_user = predictions2[idxs_user]
		for id_img in np.unique(ids_user):
			seed(0)
			idxs = sample(np.where(ids_user==id_img)[0].tolist(), 32)
			meta_vec = np.concatenate((predictions1_user[idxs, 1], predictions2_user[idxs, 1]), axis=-1)
			new_labels.append(list(labels_user[idxs[0]]))
			new_user_imgs.append(user)
			new_meta.append(list(meta_vec))
			new_ids_imgs.append(id_img)
	return new_labels, new_meta, new_user_imgs, new_ids_imgs



def get_pairs(datasetDir_train_cnn, datasetDir_val_cnn, users):
    image_paths_train = [img for user in users for img in glob.glob(datasetDir_train_cnn+user+"/*") if img[-3:] in ['jpg', 'png', 'jpeg']]
    image_paths_val = [img for user in users for img in glob.glob(datasetDir_val_cnn+user+"/*") if img[-3:] in ['jpg', 'png', 'jpeg']]
    labels_images_val = [getl_label_imgName(img) for img in image_paths_val]

    pairs = []
    for img in image_paths_train:
        labelImg = getl_label_imgName(img)
        idxs_pos_val = np.where(np.array(labels_images_val)==labelImg)[0]
        pairs_pos = [[image_paths_val[idx], img, 1, labelImg, str(getl_label_imgName(image_paths_val[idx]))+'_'+get_ID_imgName(image_paths_val[idx])] for idx in idxs_pos_val]
        seed(labelImg)
        idxs_neg_val = sample(np.where(np.array(labels_images_val)!=labelImg)[0].tolist(), len(idxs_pos_val))
        pairs_neg = [[image_paths_val[idx], img, 0, labelImg, str(getl_label_imgName(image_paths_val[idx]))+'_'+get_ID_imgName(image_paths_val[idx])] for idx in idxs_neg_val]
        pairs += pairs_pos+pairs_neg
    return pairs   


def train_meta_clf(x_train, y_train):
    print("Training meta classifier...")
    tuned_parameters = [{
        'kernel': ['rbf'], 
        'gamma': [2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1, 2, 2**2, 2**3, 2**4, 2**5, 2**6],
        'C': [2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1, 2, 2**2, 2**3, 2**4, 2**5, 2**6]
    }]      
    clf = model_selection.GridSearchCV(svm.SVC(probability=True), tuned_parameters, cv=5, scoring='roc_auc', verbose=1, n_jobs=5)
    clf.fit(x_train, y_train)
    clf.best_estimator_.fit(x_train, y_train)
    return clf


def load_models(path_models, load_meta=False):
        nbClasses = 2
        alpha = 0.5
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
        model_spec = Model(inputs=base_model1.input, outputs=output1)
        model_adt = Model(inputs=base_model2.input, outputs=output2)
        model_spec.load_weights(path_models+"/model_spectrogram.h5")
        model_adt.load_weights(path_models+"/model_adt.h5")
        if load_meta:       
            with open(path_models+"/meta_fusion.pkl", "rb") as infile: 
                clf_meta = pickle.load(infile)
        else:
            clf_meta = None
        return model_spec, model_adt, clf_meta


def get_predictions(trainDir, valDir, path_models, batch_size=512, nbClasses=2):
    datasetDir_train_cnn_spec = trainDir+"/spectrogram_multiD/"
    datasetDir_val_cnn_spec = valDir+"/spectrogram_multiD/"
    datasetDir_train_cnn_adt = trainDir+"/adt_features_multiD/"
    datasetDir_val_cnn_adt = valDir+"/adt_features_multiD/"
    users = [u for u in os.listdir(datasetDir_train_cnn_spec) if len(u.split('.'))==1]
    model_spec, model_adt, _ = load_models(path_models)
    pairs_spec = get_pairs(datasetDir_train_cnn_spec, datasetDir_val_cnn_spec, users) 
    pairs_adt = get_pairs(datasetDir_train_cnn_adt, datasetDir_val_cnn_adt, users)
    data_generator_spec = generate_batches(pairs_spec, batch_size)
    data_generator_adt = generate_batches(pairs_adt, batch_size)  
    print("Predicting train and validation sets...")
    predictions_spec = model_spec.predict_generator(data_generator_spec, steps=math.ceil(float(len(pairs_spec))/batch_size))
    predictions_adt = model_adt.predict_generator(data_generator_adt, steps=math.ceil(float(len(pairs_adt))/batch_size))
    labels_pairs = np_utils.to_categorical(np.array(pairs_spec)[:, 2], nbClasses)
    users_pairs = np.array(pairs_spec)[:, 3]
    ids_pairs = np.array(pairs_spec)[:, 4]
    return predictions_spec, predictions_adt, labels_pairs, users_pairs, ids_pairs



if __name__ == "__main__":
    """
    It trains a SVM model for a meta-classifier used to perform the late-fusion of both CNN models.

    Parameters
    ----------
        trainDir: `string` absolute path of the `train_cnn` folder to get CNN predictions from these images and use these to train the meta. 
        valDir: `string` absolute path of the `validate_cnn` folder to get CNN predictions from these images and use these to train the meta.  
        path_models: `string` absolute path of the folder where the meta-classifier model is stored.
    Output
    -------
        It stores a .pkl model in the `path_models` of the meta-classifier to perform late-fusion of both models. Also, it stores the labels, user ids and features employed in the meta training.
        
    """
    trainDir = sys.argv[1]
    valDir = sys.argv[2]
    path_models = sys.argv[3]
    predictions_spec, predictions_adt, labels_pairs, users_pairs, ids_pairs = get_predictions(trainDir, valDir, path_models)
    meta_labels, meta_features, meta_users, meta_ids = meta_predictions(labels_pairs, predictions_spec, predictions_adt, ids_pairs, users_pairs)
    clf_meta = train_meta_clf(meta_features, meta_labels)
    with open(path_models+"/meta_clf_fusion.pkl", "wb") as outfile: 
        pickle.dump(clf_meta, outfile)
    np.savetxt(path_models+"/labels_meta_clf_fusion.csv", meta_labels, delimiter=',', fmt="%d")
    np.savetxt(path_models+"/users_meta_clf_fusion.csv", np.array(meta_users).astype(int), delimiter=',', fmt="%d")
    np.save(path_models+"/features_meta_clf_fusion.npy", meta_features, allow_pickle=True)


