import os, math, pickle, random
from settings import *

def resultant_vector(train_set, test_set):
    features_train = np.sqrt(np.sum(train_set**2, axis=2))
    features_test = np.sqrt(np.sum(test_set**2, axis=2))
    return features_train, features_test


def generate_mframes_recodGait(path, subset, magnitude=False, users_valid=None):
	# return set with size (n_samples, frame_size, n_accelerations)
	print("-- LOADING DATA from "+path+subset)
	files = [path+subset+'/'+f for f in os.listdir(path+subset) if f[0] != '.']
	users = [int(f.split('/')[-1].split('___')[0].split('u')[-1]) for f in files]
	users_nsession = {u: len(np.where(users==u)[0]) for u in np.unique(users)}
	train_set = np.empty((0, frame_size, n_accelerations))
	test_set = np.empty((0, frame_size, n_accelerations))
	train_meta = np.empty((0, dim_meta), dtype=int)
	test_meta = np.empty((0, dim_meta), dtype=int)
	frame_id = 0
	for idx, f in enumerate(files):
		f_meta = f.split('___')
		user = int(f_meta[0].split('/')[-1])
		if(not users_valid or (user in users_valid)):
			session = idx+1
			data = np.loadtxt(f, delimiter=',')
			data = data[500:-500]
			frames = [data[i:i+frame_size, :] for i in range(0,data.shape[0]-frame_size,step)]
			frames = np.dstack(frames)
			frames = np.rollaxis(frames,-1)
			user_id = np.full((frames.shape[0]), user)
			session_id = np.full((frames.shape[0]), session)
			meta = np.empty((frames.shape[0], dim_meta), dtype=int)
			meta[:, label_user] = user_id
			meta[:, label_session] = session_id
			meta[:, label_frame] = np.arange(frame_id, frame_id+frames.shape[0])
			frame_id += frames.shape[0]
			if((users_nsession[user] > 3 and int(f_meta[1][-1]) > 3) or (users_nsession[user] <= 3 and int(f_meta[1][-1]) == users_nsession[user])):
				test_set = np.vstack((test_set, frames))
				test_meta = np.vstack((test_meta, meta))
			else:
				train_set = np.vstack((train_set, frames))
				train_meta = np.vstack((train_meta, meta))			
	if(magnitude):
		m_train, m_test = resultant_vector(train_set, test_set)
		train_set = np.dstack((train_set, m_train))
		test_set = np.dstack((test_set, m_test))
	return train_set, train_meta, test_set, test_meta


def generate_mframes_IDNet(path, subset, magnitude=False):
    # return set with size (n_samples, frame_size, n_accelerations)
    print("-- LOADING DATA from "+path+subset)
    files = [path+subset+'/'+f for f in os.listdir(path+subset) if f[0] != '.']
    users = [int(f.split('/')[-1].split('_')[0].split('u')[-1]) for f in files]
    users_nsession = {u: len(np.where(users==u)[0]) for u in np.unique(users)}
    train_set = np.empty((0, frame_size, n_accelerations))
    test_set = np.empty((0, frame_size, n_accelerations))
    train_meta = np.empty((0, dim_meta), dtype=int)
    test_meta = np.empty((0, dim_meta), dtype=int)
    frame_id = 0
    for idx, f in enumerate(files):
        user = users[idx]
        f_meta = f.split('/')[-1].split('_')
        walk = int(f_meta[1].split('w')[-1])
        session = idx+1
        data = np.loadtxt(f, delimiter=',')
        data = data[500:-500]
        frames = [data[i:i+frame_size, :] for i in range(0,data.shape[0]-frame_size,step)]
        frames = np.dstack(frames)
        frames = np.rollaxis(frames,-1)
        user_id = np.full((frames.shape[0]), user)
        session_id = np.full((frames.shape[0]), session)
        meta = np.empty((frames.shape[0], dim_meta), dtype=int)
        meta[:, label_user] = user_id
        meta[:, label_session] = session_id
        meta[:, label_frame] = np.arange(frame_id, frame_id+frames.shape[0])
        frame_id += frames.shape[0]
        if users_nsession[user]>1 and walk != users_nsession[user]:
                train_set = np.vstack((train_set, frames))
                train_meta = np.vstack((train_meta, meta))
        else:
                test_set = np.vstack((test_set, frames))
                test_meta = np.vstack((test_meta, meta))
    if(magnitude):
            m_train, m_test = resultant_vector(train_set, test_set)
            train_set = np.dstack((train_set, m_train))
            test_set = np.dstack((test_set, m_test))
    return train_set, train_meta, test_set, test_meta


def generate_mframes_sampling_recodGait(path, subset, train_session, test_session, rate):
    # return set with size (n_samples, frame_size, n_accelerations)
    print("-- LOADING DATA from "+path+subset)
    print(train_session, test_session)
    files = [path+subset+'/'+f for f in os.listdir(path+subset) if f[0] != '.']
    train_set = np.empty((0, frame_size, n_accelerations))
    test_set = np.empty((0, frame_size, n_accelerations))
    train_meta = np.empty((0, dim_meta), dtype=int)
    test_meta = np.empty((0, dim_meta), dtype=int)
    frame_id = 0
    for idx, f in enumerate(files):
            f_meta = f.split('___')
            user = int(f_meta[0].split('/')[-1])
            session = idx+1
            data = np.loadtxt(f, delimiter=',')
            data = data[500:-500]
            frames = [data[i:i+frame_size, :] for i in range(0,data.shape[0]-frame_size,step)]
            frames = np.dstack(frames)
            frames = np.rollaxis(frames,-1)
            if(f_meta.__contains__(train_session)):
                    n = int(frames.shape[0]*rate)
                    frames = frames[:n]
                    user_id = np.full((frames.shape[0]), user)
                    session_id = np.full((frames.shape[0]), session)
                    meta = np.empty((frames.shape[0], dim_meta), dtype=int)
                    meta[:, label_user] = user_id
                    meta[:, label_session] = session_id
                    meta[:, label_frame] = np.arange(frame_id, frame_id+frames.shape[0])
                    train_set = np.vstack((train_set, frames))
                    train_meta = np.vstack((train_meta, meta))
            elif(f.split('___').__contains__(test_session)):
                    user_id = np.full((frames.shape[0]), user)
                    session_id = np.full((frames.shape[0]), session)
                    meta = np.empty((frames.shape[0], dim_meta), dtype=int)
                    meta[:, label_user] = user_id
                    meta[:, label_session] = session_id
                    meta[:, label_frame] = np.arange(frame_id, frame_id+frames.shape[0])
                    test_set = np.vstack((test_set, frames))
                    test_meta = np.vstack((test_meta, meta))
            frame_id += frames.shape[0]
    return [train_set, train_meta, test_set, test_meta]


def sampling_negative(train_meta, klass, label, rate_neg, path_idxs, fold, train_set=None):
	print('-- SAMPLING NEGATIVE CLASS')
	uniform = True
	intercaleted = False
	shuffle = False
	directory_idxs = path_idxs+fold+'/'+str(int(klass))+'/'
	idxs_class_pos = (train_meta[ : , label] == klass).nonzero()[0]
	n_samples_neg = int(len(idxs_class_pos)*rate_neg)
	print(directory_idxs)
	if(os.path.isdir(directory_idxs)):
		print('loading indexes...')
		idxs_class_neg = np.loadtxt(directory_idxs+'idxs_neg_train.txt', dtype=int)
	else:
		idxs_class_neg_orig = (train_meta[ : , label] != klass).nonzero()[0]
		if(uniform):
			total_classes = np.unique(train_meta[:, label])
			neg_classes = np.delete(total_classes, np.where(total_classes==klass)[0])
			n_class_neg = len(neg_classes)
			n_samples_class = int(math.floor(n_samples_neg/float(len(neg_classes)) ))
			idxs_class_neg = np.hstack(([random.sample(np.where(train_meta[ : , label]==c)[0].tolist(), n_samples_class) if(len(np.where(train_meta[ : , label]==c)[0]) > n_samples_class) else list(np.where(train_meta[ : , label]==c)[0]) for c in neg_classes])).astype(int)
			idxs_class_neg = np.hstack((idxs_class_neg, np.array(random.sample(list(set(idxs_class_neg_orig)-set(idxs_class_neg)), n_samples_neg-len(idxs_class_neg)), dtype=int) ))
		else:
			idxs_class_neg = np.array(random.sample(idxs_class_neg_orig, n_samples_neg), dtype=int)
		print('saving indexes...')
		os.makedirs(directory_idxs)
		np.savetxt(directory_idxs+'idxs_neg_train.txt', idxs_class_neg, fmt='%d')
	if(intercaleted):
		idxs_final = np.array([val for pair in zip(idxs_class_pos, idxs_class_neg) for val in pair])
	else:
		idxs_final = np.hstack((idxs_class_pos, idxs_class_neg))
	if(shuffle):
		np.random.shuffle(idxs_final)
	new_train_set = train_set[idxs_final, :]
	new_train_meta = train_meta[idxs_final, :]
	return [new_train_set, new_train_meta]


if __name__ == "__main__":
    """
    It generates motion-level frames for a given set dataset and subset of data. This reads the dataset files in the folder `dataset`+`subset`, and stores a CSV file for each acceleration coordinate (x, y and z).

    Parameters
    ----------
        path_dataset: `string` absolute path of the target dataset (RecodGait v2 or IDNet's). The dataset folder should contain `recod` or `idnet` in the name.
        subset: `string` subset of the target dataset to generate the frames `user_coordinates`, `raw_data`, or `subset_train_cnn`.
        path_frames: `string` absolute path of the target dataset (RecodGait v2 or IDNet's). The dataset folder should contain recod or idnet in the name.
    Output
    -------
        It stores one CSV file for each acceleration coordinate (x, y and z) in the folder `path_frames`.
        
    """

    path_dataset = sys.argv[1]
    subset = sys.argv[2]
    path_frames = sys.argv[3]
    coordinates = {0: 'x', 1: 'y', 2: 'z'}

    if('recod' in path_dataset.lower()):
        if(len(sys.argv) > 4 and int(sys.argv[4])):
            users_traincnn = np.loadtxt(path_frames+"users_train_cnn.txt", delimiter=',', dtype="int").tolist()
            traincnn_set, traincnn_meta, validatecnn_set, validatecnn_meta = generate_mframes_recodGait(path_dataset, subset, users_valid=users_traincnn)
            if(not os.path.isdir(path_frames+"/train_cnn/")):
                os.mkdir(path_frames+"/train_cnn/")
            for i in range(traincnn_set.shape[-1]):
                np.savetxt(path_frames+"/train_cnn/frames_traincnn_"+coordinates[i]+".csv", np.hstack((traincnn_meta, traincnn_set[:, :, i])), delimiter=',', fmt="%.6f")

            if(not os.path.isdir(path_frames+"/validate_cnn/")):
                os.mkdir(path_frames+"/validate_cnn/")
            for i in range(validatecnn_set.shape[-1]):
                np.savetxt(path_frames+"/validate_cnn/frames_validatecnn_"+coordinates[i]+".csv", np.hstack((validatecnn_meta, validatecnn_set[:, :, i])), delimiter=',', fmt="%.6f")

        else:
            users_validatecnn = np.loadtxt(path_frames+"users_validate_cnn.txt", delimiter=',', dtype="int").tolist()
            gallery_set, gallery_meta, probe_set, probe_meta = generate_mframes_recodGait(path_dataset, subset, users_valid=users_validatecnn)
            if(not os.path.isdir(path_frames+"/gallery/")):
                os.mkdir(path_frames+"/gallery/")
            for i in range(gallery_set.shape[-1]):
                np.savetxt(path_frames+"/gallery/frames_gallery_"+coordinates[i]+".csv", np.hstack(( gallery_meta,  gallery_set[:, :, i])), delimiter=',', fmt="%.6f")

            if(not os.path.isdir(path_frames+"/probe/")):
                os.mkdir(path_frames+"/probe/")
            for i in range(probe_set.shape[-1]):
                np.savetxt(path_frames+"/probe/frames_probe_"+coordinates[i]+".csv", np.hstack((probe_meta, probe_set[:, :, i])), delimiter=',', fmt="%.6f")

    elif('idnet' in path_dataset.lower()):
        gallery_set, gallery_meta, probe_set, probe_meta = generate_mframes_IDNet(path_dataset, subset)
        if(not os.path.isdir(path_frames+"/gallery/")):
            os.mkdir(path_frames+"/gallery/")
        for i in range(gallery_set.shape[-1]):
            np.savetxt(path_frames+"/gallery/frames_gallery_"+coordinates[i]+".csv", np.hstack(( gallery_meta,  gallery_set[:, :, i])), delimiter=',', fmt="%.6f")

        if(not os.path.isdir(path_frames+"/probe/")):
            os.mkdir(path_frames+"/probe/")
        for i in range(probe_set.shape[-1]):
            np.savetxt(path_frames+"/probe/frames_probe_"+coordinates[i]+".csv", np.hstack((probe_meta, probe_set[:, :, i])), delimiter=',', fmt="%.6f")

    else:
        print("Invalid dataset!")

