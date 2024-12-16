import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pre_process(folder_path, coke = 1):

    files = []

    for filename in os.listdir(folder_path):
        if filename[:-4] not in files:
            files.append(filename[:-4])

    amp = []
    auc = []
    frq = []

    y = []


    for file in files:

        npz = np.load(folder_path+'/'+file+'.npz')
        csv = pd.read_csv(folder_path+'/'+file+'.csv', usecols=['ALP', 'ALP_Prev'])

        alp = csv['ALP'].to_numpy().reshape(-1, 1)
        alp_prev = csv['ALP_Prev'].to_numpy().reshape(-1, 1)

        amp_zz = npz['amp_zz']
        auc_zz = npz['auc_zz']
        frq_zz = npz['frq_zz']

        if coke:
            label = np.ones(len(csv)).reshape(-1, 1)
        else:
            label = np.zeros(len(csv)).reshape(-1, 1)

        day = file[-4:-2]
        sess = file[-2:]

        if day == 'D1' or day =='D2':
            y_day = np.zeros(len(csv)).reshape(-1, 1)
        else:
            y_day = np.ones(len(csv)).reshape(-1, 1)

        if sess == 'S1':
            y_sess = np.zeros(len(csv)).reshape(-1, 1)
        else:
            y_sess = np.ones(len(csv)).reshape(-1, 1)

        amp.append(amp_zz)
        auc.append(auc_zz)
        frq.append(frq_zz)
        
        y.append(np.concatenate([alp, alp_prev, label, y_day, y_sess], axis =1))

    amp = np.vstack(amp)
    auc = np.vstack(auc)
    frq = np.vstack(frq)

    y = np.vstack(y)

    return amp, auc, frq, files, y


def extract_principal_vec_pca(data):

    pca = PCA()

    pca.fit(data)
   
    var = pca.explained_variance_
    
    com = pca.components_

    val_sum = sum(pca.explained_variance_)

    sort_ind = np.argsort(var)
    sort_ind = sort_ind[::-1]

    temp_sum = 0
    principle_vec = []
    principle_val = []
    i=0
    while(temp_sum < 0.95*val_sum):
        principle_vec.append(com[sort_ind[i]])
        principle_val.append(var[sort_ind[i]])
        temp_sum += var[sort_ind[i]]
        i += 1
    
    principle_vec = np.matrix(principle_vec)

    return principle_vec, i

def pca_transform(pca_vec, data):

    transform_data = np.dot(data, pca_vec.T)

    return transform_data



def pca(train_data, test_data):

    pca_vec, pca_comp = extract_principal_vec_pca(train_data)

    train_transform = pca_transform(pca_vec, train_data)
    test_transform = pca_transform(pca_vec, test_data)

    return pca_vec, pca_comp, train_transform, test_transform

def process_data(folder_path_train, folder_path_test, group):
    
    amp_train, auc_train, frq_train, files_train, y_train = pre_process(folder_path_train, group)
    amp_test, auc_test, frq_test, files_test, y_test = pre_process(folder_path_test, group)

    return amp_train, auc_train, frq_train, amp_test, auc_test, frq_test, files_train, files_test, y_train, y_test

if __name__ == '__main__':

    folder_path_coke_train = '15SEC/COKE/TRAIN/resnet152'
    folder_path_sal_train = '15SEC/SAL/TRAIN/resnet152'

    folder_path_coke_test = '15SEC/COKE/TEST/resnet152'
    folder_path_sal_test = '15SEC/SAL/TEST/resnet152'

    amp_train, auc_train, frq_train, amp_test, auc_test, frq_test, files_train, files_test, y_train, y_test = process_data(folder_path_coke_train, folder_path_coke_test, 1)

    amp_pca_vec, amp_comp, amp_train_transform, amp_test_transform = pca(amp_train, amp_test)
    auc_pca_vec, auc_comp, auc_train_transform, auc_test_transform = pca(auc_train, auc_test)
    frq_pca_vec, frq_comp, frq_train_transform, frq_test_transform = pca(frq_train, frq_test)

    np.savez('15SEC_coke_pca.npz', 
            amp_train=amp_train_transform, amp_test=amp_test_transform,
            auc_train=auc_train_transform, auc_test=auc_test_transform,
            frq_train=frq_train_transform, frq_test=frq_test_transform,
            y_train=y_train, y_test=y_test, files_train=files_train, files_test=files_test,
            amp_pca_vec=amp_pca_vec, auc_pca_vec=auc_pca_vec, frq_pca_vec=frq_pca_vec,
            amp_comp=amp_comp, auc_comp=auc_comp, frq_comp=frq_comp)
    
    

    amp_train, auc_train, frq_train, amp_test, auc_test, frq_test, files_train, files_test, y_train, y_test = process_data(folder_path_sal_train, folder_path_sal_test, 0)

    amp_pca_vec, amp_comp, amp_train_transform, amp_test_transform = pca(amp_train, amp_test)
    auc_pca_vec, auc_comp, auc_train_transform, auc_test_transform = pca(auc_train, auc_test)
    frq_pca_vec, frq_comp, frq_train_transform, frq_test_transform = pca(frq_train, frq_test)

    np.savez('15SEC_sal_pca.npz', 
            amp_train=amp_train_transform, amp_test=amp_test_transform,
            auc_train=auc_train_transform, auc_test=auc_test_transform,
            frq_train=frq_train_transform, frq_test=frq_test_transform,
            y_train=y_train, y_test=y_test, files_train=files_train, files_test=files_test,
            amp_pca_vec=amp_pca_vec, auc_pca_vec=auc_pca_vec, frq_pca_vec=frq_pca_vec,
            amp_comp=amp_comp, auc_comp=auc_comp, frq_comp=frq_comp)


    amp_coke_train, auc_coke_train, frq_coke_train, amp_coke_test, auc_coke_test, frq_coke_test, files_coke_train, files_coke_test, y_coke_train, y_coke_test = process_data(folder_path_coke_train, folder_path_coke_test, 1)
    amp_sal_train, auc_sal_train, frq_sal_train, amp_sal_test, auc_sal_test, frq_sal_test, files_sal_train, files_sal_test, y_sal_train, y_sal_test = process_data(folder_path_sal_train, folder_path_sal_test, 0)

    amp_train = np.vstack((amp_coke_train, amp_sal_train))
    auc_train = np.vstack((auc_coke_train, auc_sal_train))
    frq_train = np.vstack((frq_coke_train, frq_sal_train))

    files_train = files_coke_train + files_sal_train

    y_train =  np.vstack((y_coke_train, y_sal_train))

    amp_test = np.vstack((amp_coke_test, amp_sal_test))
    auc_test = np.vstack((auc_coke_test, auc_sal_test))
    frq_test = np.vstack((frq_coke_test, frq_sal_test))

    files_test = files_coke_test + files_sal_test

    y_test =  np.vstack((y_coke_test, y_sal_test))


    amp_pca_vec, amp_comp, amp_train_transform, amp_test_transform = pca(amp_train, amp_test)
    auc_pca_vec, auc_comp, auc_train_transform, auc_test_transform = pca(auc_train, auc_test)
    frq_pca_vec, frq_comp, frq_train_transform, frq_test_transform = pca(frq_train, frq_test)

    np.savez('15SEC_coke_sal_pca.npz', 
            amp_train=amp_train_transform, amp_test=amp_test_transform,
            auc_train=auc_train_transform, auc_test=auc_test_transform,
            frq_train=frq_train_transform, frq_test=frq_test_transform,
            y_train=y_train, y_test=y_test, files_train=files_train, files_test=files_test,
            amp_pca_vec=amp_pca_vec, auc_pca_vec=auc_pca_vec, frq_pca_vec=frq_pca_vec,
            amp_comp=amp_comp, auc_comp=auc_comp, frq_comp=frq_comp)