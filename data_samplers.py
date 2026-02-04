import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.spatial.distance import cdist

def weights_closest(pi):
    weights = ((pi-pi.min())==0).squeeze()
    return weights/weights.sum()

def weights_ponderate(pi):
    weights = 1/(pi-pi.min()+1).squeeze()**2
    return weights/weights.sum()


class BYOLSupDataset(Dataset):
    def __init__(self, tags_file, img_data, transform=None, target_transform=None,
                 weightfunc=weights_closest):
        self.all_labels = pd.read_csv(tags_file)
        self.img_data = img_data
        self.transform = transform
        self.target_transform = target_transform
        self.weightfunc = weightfunc

    def __len__(self):
        return len(self.all_labels.shape[0])

    def __getitem__(self, idx):
        fid_ID = self.all_labels.iloc[idx,:]
        all_tags_nofid = self.all_tags.drop(rd_idx)
        pi = cdist(fid_label, all_tags_nofid, metric='cityblock')
        weights = self.weightfunc(pi)
        sample = np.random.choice(all_tags_nofid.shape[0], p=weights)
        idx_friend = np.atleast_1d(all_tags_nofid.index[sample])
        mdist = pi[sample]
        return img_data[idx], img_data[idx_friend], mdist


        
