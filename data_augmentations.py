import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate, ElasticTransform
from sklearn.model_selection import KFold

sep = "\\"
master_path = "C:"+sep+"Baseline_Dataset"+sep+"dataset_fix"+sep
H = 256
W = 256

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    """ Loading the images and masks """
    
    X = sorted(glob(os.path.abspath(os.path.join(path, "images"+sep+"*", "*.png"))))
    Y = sorted(glob(os.path.abspath(os.path.join(path, "masks_denormalized"+sep+"*", "*.png"))))

    """ Spliting the data into training and testing """
    split_size = int(len(X) * split)

    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)
    
    return (train_x, train_y), (test_x, test_y)


def augment_data(domain, images, masks, save_path, augment=True):

    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        if domain=="radiography":
            name = x.split(sep)[-2] + "_" + x.split(sep)[-1].split(".")[0]
        elif domain=="busi":
            name = x.split(sep)[-2] + "_" + x.split(sep)[-1].split(".")[0]

        """ Reading the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=1, sigma=50, alpha_affine=50)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            aug = OpticalDistortion(p=1, distort_limit=0.05, shift_limit=0.05)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5, x6]
            Y = [y, y1, y2, y3, y4, y5, y6]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            try:
                aug = cv2.resize(i, (W, H))
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the dataset """
    data_path = "C:\\Baseline_Dataset\\BUSI\\"
    domain_dataset = "busi"
    master_folder = "new_data_hybrid_(80_20)_20_2class_denom_"
    
    (train_x, train_y), (test_x, test_y) = load_data(data_path, 0.2)

    print(f"Train:\t {len(train_x)} - {len(train_y)}")
    print(f"Test:\t {len(test_x)} - {len(test_y)}")

    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/test/image/")
    create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/test/mask/")

    augment_data(domain_dataset, test_x, test_y, master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/test/", augment=False)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    for train_idx, val_idx in kfold.split(train_x):
        fold += 1
        print(f"Training fold {fold}")

        train_x_fold, valid_x_fold = train_x[train_idx], train_x[val_idx]
        train_y_fold, valid_y_fold = train_y[train_idx], train_y[val_idx]

        print("Train Fold")
        print(len(train_x_fold))
        print(len(train_y_fold))

        print("Valid Fold")
        print(len(valid_x_fold))
        print(len(valid_y_fold))
    
        """ Create directories to save the augmented data """
        create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/train/image/")
        create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/train/mask/")
        create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/image/")
        create_dir(master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/mask/")

        """ Data augmentation """
        augment_data(domain_dataset, train_x_fold, train_y_fold, master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/train/", augment=True)
        augment_data(domain_dataset, valid_x_fold, valid_y_fold, master_path+master_folder+domain_dataset+"_fold_"+str(W)+"/fold_"+str(fold)+"/valid/", augment=False)
