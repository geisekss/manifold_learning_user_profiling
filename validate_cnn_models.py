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
from keras.backend.tensorflow_backend import set_session
from contextlib import redirect_stdout
from sklearn import metrics



def getl_label_imgName(imgName):
    return int(imgName.split("/")[-2])


def get_ID_imgName(imgName):
    return imgName.split("_")[-1].split('.')[0]


def get_pairs_user(user, testDir, labelsImagesGallery, imagesGallery):
        filesUser = [img for img in glob.glob(testDir+user+"/*") if img[-3:] in ['jpg', 'png', 'jpeg']]
        labelsImagesGallery = np.array(labelsImagesGallery)
        idxs_pos = np.where(np.array(labelsImagesGallery)==int(user))[0]
        pairs = []
        for f in filesUser:
                idImage = get_ID_imgName(f)
                pairs_pos = [[f, imagesGallery[idx], 1, idImage] for idx in idxs_pos]
                pairs += pairs_pos
        filesNeg = np.array([img for u_neg in os.listdir(testDir) if u_neg != user for img in glob.glob(testDir+u_neg+"/*") if img[-3:] in ['jpg', 'png', 'jpeg']])
        labelsTestNeg = np.array([getl_label_imgName(img) for img in filesNeg])
        idxs_neg = sampling_uniform(labelsTestNeg, user, len(idxs_pos)*5)
        for f in filesNeg[idxs_neg]:
                idImage = get_ID_imgName(f)
                pairs_neg = [[f, imagesGallery[idx], 0, idImage] for idx in idxs_pos]
                pairs += pairs_neg
        return pairs


def load_test_batches(user, testDir_spec, galleryDir_spec, batch_size=128, nbClasses=2):
        files_gallery = [img for img in glob.glob(galleryDir_spec+"*/*") if img[-3:] in ['jpg', 'png', 'jpeg']]
        labels_gallery = [getl_label_imgName(img) for img in files_gallery]

        batch1, batch2, labelList, idsList, usersImgs = [], [], [], [], []
        nInBatch = 0
        pairsUser = get_pairs_user(user, testDir_spec, labels_gallery, files_gallery)

        for i in range(len(pairsUser)):
                img_a1 = img_to_array(load_img(pairsUser[i][0], target_size=(224, 224)))
                img_a2 = img_to_array(load_img(pairsUser[i][1], target_size=(224, 224)))
                img_a = np.concatenate((img_a1, img_a2), axis=2)
                img_b1 = img_to_array(load_img(pairsUser[i][0].replace('spectrogram_multiD', 'adt_features_multiD'), target_size=(224, 224)))
                img_b2 = img_to_array(load_img(pairsUser[i][1].replace('spectrogram_multiD', 'adt_features_multiD'), target_size=(224, 224)))
                img_b = np.concatenate((img_b1, img_b2), axis=2)

                userImg = pairsUser[i][2]
                label = np_utils.to_categorical(userImg, nbClasses)
                id = pairsUser[i][3]
        
                batch1.append(img_a)
                batch2.append(img_b)
                labelList.append(label)
                idsList.append(id)
                usersImgs.append(userImg)
                nInBatch += 1
                
                #if we already have one batch, yields it
                if nInBatch >= batch_size:
                        yield [np.array(batch1), np.array(batch2)], np.array(labelList), np.array(idsList), np.array(usersImgs)
                        batch1, batch2, labelList, idsList, usersImgs = [], [], [], [], []
                        nInBatch = 0

        #yield the remaining of the batch
        if nInBatch > 0:
                yield [np.array(batch1), np.array(batch2)], np.array(labelList), np.array(idsList), np.array(usersImgs)


def load_models(path_models, nbClasses=2, alpha=0.5, load_meta=False):
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
                with open(path_models+"/meta_clf_fusion.pkl", "rb") as infile: 
                        clf_meta = pickle.load(infile)
        else:
                clf_meta = None
        return model_spec, model_adt, clf_meta


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



if __name__ == "__main__":
    """
    Validates the CNN model and meta-classifier models by testing these models using the gallery and probe images.

    Parameters
    ----------
        galleryDir: `string` absolute path of the folder where the user gallery images (`gallery` fold) are stored. 
        testDir: `string` absolute path of the folder where the user probe images (`probe` fold) are stored. 
        path_models: `string` absolute path of the folder where the CNN, and meta-classifier models are stored.
    Output
    -------
        It stores JSON files in the `path_models` with the results obtained from the Spectrogram CNN, ADT CNN and late-fusion models.
        
    """

    galleryDir = sys.argv[1]
    testDir = sys.argv[2]
    path_models = sys.argv[3]

    model_spec, model_adt, clf_meta = load_models(path_models, load_meta=True)
    results = {}
    results_spec = {}
    results_adt = {}

    testDir_spec = testDir+"/spectrogram_multiD/"
    users_test = [user for user in os.listdir(testDir_spec) if user[0]!='.' and len(user.split('.'))==1]
    print("Evaluating each user ...")
    for u in users_test:
            print(u)
            test_cnn_generator = load_test_batches(u, testDir_spec, galleryDir+"/spectrogram_multiD/", batch_size=1024)
            total_predictions_spec = np.empty((0, 2))
            total_predictions_adt = np.empty((0, 2))
            total_labels = np.empty((0, 2))
            total_ids = np.empty((0,))
            for batch, labels, ids, userImgs in test_cnn_generator:
                    predictions1 = model_spec.predict_on_batch(batch[0])
                    predictions2 = model_adt.predict_on_batch(batch[1])
                    total_predictions_spec = np.concatenate((total_predictions_spec, predictions1))
                    total_predictions_adt = np.concatenate((total_predictions_adt, predictions2))
                    total_labels = np.concatenate((total_labels, labels))
                    total_ids = np.concatenate((total_ids, ids))

            features_u = np.hstack((total_predictions_spec, total_predictions_adt))
            frames_u = []
            ids_frames = np.empty((0,))
            users_frames = np.empty((0,))
            labels_frames = np.empty((0,2))
            for id in np.unique(total_ids):
                idxs_u_id = np.where(total_ids==id)[0]
                frames_u_id = [features_u[idxs_u_id][k:k+32] if(k+32<len(idxs_u_id)) else features_u[idxs_u_id][len(idxs_u_id)-32:] for k in range(0, len(idxs_u_id), 24)]
                frames_u += frames_u_id
                ids_frames = np.concatenate((ids_frames, np.repeat(id, len(frames_u_id)) ))
                users_frames = np.concatenate((users_frames, np.repeat(u, len(frames_u_id)) ))
                labels_frames = np.concatenate((labels_frames, np.repeat([total_labels[idxs_u_id[0]]], len(frames_u_id), axis=0) ))
            frames_u = np.dstack((np.array(frames_u)[:, :, 1], np.array(frames_u)[:, :, -1])).reshape((len(frames_u), -1))
            preds_frames = clf_meta.best_estimator_.predict_proba(frames_u)
            final_labels, final_predictions = average_fusion(labels_frames, preds_frames, ids_frames)
            idxs_pos = np.where(np.array(final_labels)[:, 1]==1)[0]
            idxs_neg = np.where(np.array(final_labels)[:, 1]==0)[0]
            fusion_labels, fusion_predictions = max_prob_temporal(np.concatenate((np.array(final_labels)[idxs_pos], np.array(final_labels)[idxs_neg])), np.concatenate((np.array(final_predictions)[idxs_pos], np.array(final_predictions)[idxs_neg])))
            results[u] = [np.average(metrics.recall_score(np.argmax(labels_frames, axis=1), np.argmax(preds_frames, axis=1), average=None)), np.average(metrics.recall_score(np.argmax(final_labels, axis=1), np.argmax(final_predictions, axis=1), average=None)), np.average(metrics.recall_score(np.argmax(fusion_labels, axis=1), np.argmax(fusion_predictions, axis=1), average=None)) ]
            print("Fusion")
            print("acc norm meta:", results[u][0], "| acc norm average:", results[u][1], "| acc norm fusion:", results[u][2])
            final_labels, final_predictions = average_fusion(total_labels, total_predictions_spec, total_ids)
            idxs_pos = np.where(np.array(final_labels)[:, 1]==1)[0]
            idxs_neg = np.where(np.array(final_labels)[:, 1]==0)[0]
            fusion_labels, fusion_predictions = max_prob_temporal(np.concatenate((np.array(final_labels)[idxs_pos], np.array(final_labels)[idxs_neg])), np.concatenate((np.array(final_predictions)[idxs_pos], np.array(final_predictions)[idxs_neg])))
            results_spec[u] = [np.average(metrics.recall_score(np.argmax(total_labels, axis=1), np.argmax(total_predictions_spec, axis=1), average=None)),  np.average(metrics.recall_score(np.argmax(final_labels, axis=1), np.argmax(final_predictions, axis=1), average=None)), np.average(metrics.recall_score(np.argmax(fusion_labels, axis=1), np.argmax(fusion_predictions, axis=1), average=None)) ]
            print("Spectrogram")
            print("acc norm total:", results_spec[u][0], "| acc norm voting:", results_spec[u][1], "| acc norm fusion:", results_spec[u][2])
            final_labels, final_predictions = average_fusion(total_labels, total_predictions_adt, total_ids)
            idxs_pos = np.where(np.array(final_labels)[:, 1]==1)[0]
            idxs_neg = np.where(np.array(final_labels)[:, 1]==0)[0]
            fusion_labels, fusion_predictions = max_prob_temporal(np.concatenate((np.array(final_labels)[idxs_pos], np.array(final_labels)[idxs_neg])), np.concatenate((np.array(final_predictions)[idxs_pos], np.array(final_predictions)[idxs_neg])))
            results_adt[u] = [np.average(metrics.recall_score(np.argmax(total_labels, axis=1), np.argmax(total_predictions_adt, axis=1), average=None)),  np.average(metrics.recall_score(np.argmax(final_labels, axis=1), np.argmax(final_predictions, axis=1), average=None)), np.average(metrics.recall_score(np.argmax(fusion_labels, axis=1), np.argmax(fusion_predictions, axis=1), average=None)) ]
            print("ADT")
            print("acc norm total:", results_adt[u][0], "| acc norm voting:", results_adt[u][1], "| acc norm fusion:", results_adt[u][2])
    

    json_results_spec = {'results': np.array([np.mean(list(results_spec.values()), axis=0), np.std(list(results_spec.values()), axis=0)]).T.tolist(), 'scores': results_spec}
    json_results_adt = {'results': np.array([np.mean(list(results_adt.values()), axis=0), np.std(list(results_adt.values()), axis=0)]).T.tolist(), 'scores': results_adt}
    json_results_fusion = {'results': np.array([np.mean(list(results.values()), axis=0), np.std(list(results.values()), axis=0)]).T.tolist(), 'scores': results}
    with open(path_models+"/results_average_fusion_model_spectrogram.json", "w") as outfile: 
            json.dump(json_results_spec, outfile, indent=1)
    with open(path_models+"/results_average_fusion_model_adt.json", "w") as outfile: 
            json.dump(json_results_adt, outfile, indent=1)
    with open(path_models+"/results_meta_fusion_models_spec_adt.json", "w") as outfile: 
            json.dump(json_results_fusion, outfile, indent=1)


