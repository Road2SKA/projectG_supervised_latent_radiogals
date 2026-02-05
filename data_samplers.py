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
    def __init__(self, 
                 tags_file, 
                 img_data, 
                 transform=None, 
                 friend_transform=None,
                 weightfunc=weights_closest,
                 p_pair_from_class=0.5):
        self.all_labels = pd.read_csv(tags_file)
        self.img_data = img_data
        self.transform = transform
        self.friend_transform = friend_transform
        self.weightfunc = weightfunc
        self.p_pair_from_class = p_pair_from_class

    def __len__(self):
        return len(self.all_labels.shape[0])

    def __getitem__(self, idx):
        img = self.img_data[idx]
        label_vec = self.all_labels.iloc[idx, :].values.reshape(1, -1)

        u = np.random.rand()
        if u < self.p_pair_from_class:
            all_tags_nofid = self.all_labels.drop(index=idx)
            pi = cdist(label_vec, all_tags_nofid.values, metric="cityblock")
            weights = self.weightfunc(pi)
            sample = np.random.choice(all_tags_nofid.shape[0], p=weights)
            idx_friend = all_tags_nofid.index[sample]
            mdist = pi[0, sample]

            img_friend = self.img_data[idx_friend]
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            if self.friend_transform:
                img_friend = self.friend_transform(img_friend)
        else:
            img_friend = img  # same underlying image
            mdist = 0.0       # distance to itself

            if self.transform:
                img = self.transform(img)

            # MUST be augmented differently
            if self.friend_transform:
                img_friend = self.friend_transform(img_friend)
            
        return img, img_friend, mdist


        
