import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix 
from metrics import dice_loss, dice_coef, iou, focal_loss_non_weight, logcoshDice, auc
from train import load_data
from config import get_config
import tensorflow_addons as tfa
import csv
from sklearn.metrics import classification_report

""" Global parameters """
# H = 224
# W = 224

sep = "\\"
master_dataset = "C:"+sep+"Baseline_Dataset"+sep+"dataset_fix"+sep
master_path = ""

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y1 = sorted(glob(os.path.join(path, "mask", "*png")))
    y2 = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y1, y2

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def read_label(path):
    path = path.decode()
    if domain == "radiography":
        x = (path.split(sep)[-1]).split("_")[0]
    elif domain == "busi":
        x = (path.split(sep)[-1]).split("_")[0]
    
    x = cfg["class_names"].index(x)
    x = np.array(x, dtype=np.int32)

    return x

def tf_parse(x, y1, y2):
    def _parse(x, y1, y2):
        x = read_image(x)
        y1 = read_mask(y1)
        y2 = read_label(y2)
        return x, y1, y2

    x, y1, y2 = tf.numpy_function(_parse, [x, y1, y2], [tf.float32, tf.float32, tf.int32])
    y2 = tf.one_hot(y2, cfg["num_classes"])

    x.set_shape([cfg["image_size"], cfg["image_size"], 3])
    y1.set_shape([cfg["image_size"], cfg["image_size"], 1])
    y2.set_shape(cfg["num_classes"])

    return x, y1, y2

def tf_dataset(X, Y1, Y2, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X,  Y1, Y2))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

def save_results(image, mask, y1_pred, save_image_path):
    ## i - m - yp - yp*i
    line = np.ones((cfg["image_size"], 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    mask = mask * 255

    y1_pred = np.expand_dims(y1_pred, axis=-1)    ## (512, 512, 1)
    y1_pred = np.concatenate([y1_pred, y1_pred, y1_pred], axis=-1)  ## (512, 512, 3)

    masked_image = image * y1_pred
    y1_pred = y1_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y1_pred, line, masked_image], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """Load Config Hyperparameter"""
    cfg = get_config()

    dropout_rate = 0.3
    fold_setting = 1 #setting 1 to 5

    domain = "radiography"
    dataset_path = master_dataset+"new_data_hybrid_(80_20)_20_2class_denom_"+domain+"_fold_"+str(cfg["image_size"])
    name_model = cfg["name_model_a"]

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading model """
    custom_objects = {'iou': iou, 'dice_coef': dice_coef, 'logcoshDice': logcoshDice, 'focal_loss_fixed': focal_loss_non_weight(alpha=cfg["alpha_focal_loss"], gamma=cfg["gamma_focal_loss"])}
    model = tf.keras.models.load_model("files"+sep+domain+sep+"model_"+domain+"_"+name_model+"_"+str(fold_setting)+"_"+str(dropout_rate)+".h5", custom_objects=custom_objects)


    """ Load the dataset """
    test_path = os.path.join(dataset_path, "test")
    test_x, test_y1, test_y2 = load_data(test_path)
    print(f"Test: {len(test_x)} - {len(test_y1)} - {len(test_y2)}")
    test_dataset = tf_dataset(test_x, test_y1, test_y2, batch=cfg["batch_size"])
    test_ds = test_dataset.map(lambda x, y, z: (x, (y, z)))

    """ Evaluation and Prediction """
    SCORE = [] #segmentation
    SCORE_2 = [] #classification
    y_true_list = []
    y_pred_list = []

    num_test = len(test_x)

    images_test = []
    masks_test = []
    preds_test = []

    y_true_test = []
    preds_class_test = []

    # Calculate performance metrics
    dsc_sc = np.zeros((num_test,1))
    iou_sc = np.zeros_like(dsc_sc)
    rec_sc = np.zeros_like(dsc_sc)
    tn_sc = np.zeros_like(dsc_sc)
    prec_sc = np.zeros_like(dsc_sc)
    f1_sc = np.zeros_like(dsc_sc)

    # Calculate performance metrics
    dsc_sc_c = np.zeros((num_test,1))
    iou_sc_c = np.zeros_like(dsc_sc_c)
    rec_sc_c = np.zeros_like(dsc_sc_c)
    tn_sc_c = np.zeros_like(dsc_sc_c)
    prec_sc_c = np.zeros_like(dsc_sc_c)
    f1_sc_c = np.zeros_like(dsc_sc_c)



    # for x, y1, y2 in tqdm(zip(test_x, test_y1, test_y2), total=len(test_x)):
    with open('results'+sep+domain+sep+'output_class_predict.csv', mode='w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['Example', 'Class', 'Class Name'])
        for i, (x, y1, y2) in tqdm(enumerate(zip(test_x, test_y1, test_y2)), total=len(test_x)):
            """ Extract the name """
            name = "example_" + str(i+1) + "_" + x.split(sep)[-1].split(".")[0]

            """ Reading the image """
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            x = image/255.0
            x = np.expand_dims(x, axis=0)

            """ Reading the mask """
            mask = cv2.imread(y1, cv2.IMREAD_GRAYSCALE)
            mask = mask/255.0
            mask = mask.astype(np.int32)
            # mask = np.expand_dims(mask, axis=-1)

            """ Reading the class label """
            if domain == "radiography":
                y_true = (y2.split(sep)[-1]).split("_")[0]
            elif domain == "busi":
                y_true = (y2.split(sep)[-1]).split("_")[0]

            y_true = cfg["class_names"].index(y_true)
            y_true_list.append(y_true)
            y_true = np.array([y_true])

            if name_model!= "only_classification":
                """ Prediction """
                # Split the results into segmentation and classification outputs
                y1_pred, y2_pred = model.predict(x)

                # y1_pred = np.squeeze(y1_pred, axis=-1)
                y1_pred = np.squeeze(y1_pred, axis=0)
                y1_pred = np.squeeze(y1_pred, axis=-1)
                y1_pred = y1_pred > 0.5
                y1_pred = y1_pred.astype(np.int32)
                

                """ Saving the prediction """
                save_image_path = f"results{sep}{domain}{sep}{name}.png"
                save_results(image, mask, y1_pred, save_image_path)

                """ Flatten the array """
                mask = mask.flatten()
                y1_pred = y1_pred.flatten()
                
                """ Calculating the metrics values augmentation"""
                scores = auc(mask, y1_pred)
                dsc_sc[i], iou_sc[i], rec_sc[i], prec_sc[i], f1_sc[i]= scores

                images_test.append(x)
                masks_test.append(mask)
                preds_test.append(y1_pred)

                """ Calculating the metrics values """
                acc_value = accuracy_score(mask, y1_pred)
                f1_value = f1_score(mask, y1_pred, labels=[0, 1], average="binary")
                jac_value = jaccard_score(mask, y1_pred, labels=[0, 1], average="binary")
                recall_value = recall_score(mask, y1_pred, labels=[0, 1], average="binary")
                precision_value = precision_score(mask, y1_pred, labels=[0, 1], average="binary")
                SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])
            else:
                # Split the results into segmentation and classification outputs
                y2_pred = model.predict(x)

            # Get the predicted classes from the classification output
            y2_pred_class = np.argmax(y2_pred, axis=1)
            y_pred_list.append(y2_pred_class[0])

            y_true_test.append(y_true)
            preds_class_test.append(y2_pred_class)

            scores_2 = auc(y_true, y2_pred_class)
            _, _, rec_sc_c[i], prec_sc_c[i], f1_sc_c[i]= scores_2

            # Print the predicted classes for the first few examples
            y2_pred_class_names = cfg["class_names"][y2_pred_class[0]]
            text_output = f"Example {i+1}: Class {y2_pred_class[0]} - {y2_pred_class_names}"
            writer.writerow([i+1, y2_pred_class[0], y2_pred_class_names])

            """ Calculating the metrics values of the classification output"""
            acc_value_2 = accuracy_score(y_true, y2_pred_class)
            f1_value_2 = f1_score(y_true, y2_pred_class, average="weighted")
            recall_value_2 = recall_score(y_true, y2_pred_class, average="weighted")
            precision_value_2 = precision_score(y_true, y2_pred_class, average="weighted")
            SCORE_2.append([name, acc_value_2, f1_value_2, recall_value_2, precision_value_2])

        print(classification_report(y_true_list, y_pred_list, target_names=cfg["class_names"]))

        # Confusion matrix
        masks_test = np.array(masks_test)
        preds_test = np.array(preds_test)
        confusion = confusion_matrix(masks_test.ravel(),preds_test.ravel()>0.5)
        # print(confusion)
        accuracy = 0
        if float(np.sum(confusion))!=0:
            accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
        print(' Global Acc \t{0:^.4f}'.format(accuracy))

        print('\n DSC \t\t{0:^.4f} \n IOU \t\t{1:^.4f}\n F1 \t\t{3:^.4f} \n Recall \t{2:^.4f} \n Precision\t{3:^.4f}'.format(
            np.sum(dsc_sc)/num_test,  
            np.sum(iou_sc)/num_test, 
            np.sum(f1_sc)/num_test, 
            np.sum(rec_sc)/num_test,
            np.sum(prec_sc)/num_test ))

        # Area under the ROC curve
        AUC_ROC = roc_auc_score(preds_test.ravel()>0.5, masks_test.ravel())
        print(' AUC ROC \t{0:^.4f}'.format(AUC_ROC))

        print('\n')
        print('*'*60)

        #========================

        y_true_test = np.array(y_true_test)
        preds_class_test = np.array(preds_class_test)

        confusion = confusion_matrix(y_true_test.ravel(),preds_class_test.ravel())
        # print(confusion)
        accuracy = 0
        if float(np.sum(confusion))!=0:
            #change this one to suit the number of class
            if domain == "radiography":
                accuracy = float(confusion[0,0]+confusion[1,1]+confusion[2,2]+confusion[3,3])/float(np.sum(confusion)) #for radiography
            elif domain == "busi":
                accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion)) #for busi            

        print(' Global Acc \t{0:^.4f}'.format(accuracy))

        print('\n F1 \t\t{2:^.4f} \n Recall \t{0:^.4f} \n Precision\t{1:^.4f}'.format(
            # np.sum(dsc_sc_c)/num_test,  
            # np.sum(iou_sc_c)/num_test,  
            np.sum(f1_sc_c)/num_test,
            np.sum(rec_sc_c)/num_test,
            np.sum(prec_sc_c)/num_test ))
        
        F1Score = f1_score(y_true_test.ravel(), preds_class_test.ravel(), average="weighted")
        print(' FI-Score \t{0:^.4f}'.format(F1Score))

        RecallScore = recall_score(y_true_test.ravel(), preds_class_test.ravel(), average="weighted")
        print(' Recall \t{0:^.4f}'.format(RecallScore))

        PrecisionScore = precision_score(y_true_test.ravel(), preds_class_test.ravel(), average="weighted")
        print(' Precision \t{0:^.4f}'.format(PrecisionScore))

        print('\n')
        print('*'*60)


    if name_model!= "only_classification":
        """ Metrics values """
        print("==Metrics values Segmentation==")
        score = [s[1:]for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"Accuracy: {score[0]:0.4f}")
        print(f"F1: {score[1]:0.4f}")
        print(f"Jaccard: {score[2]:0.4f}")
        print(f"Recall: {score[3]:0.4f}")
        print(f"Precision: {score[4]:0.4f}")

        df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
        df.to_csv("files"+sep+domain+sep+"score_"+domain+"_"+name_model+"_"+str(dropout_rate)+".csv")


    """ Metrics values Classification"""
    print("==Metrics values Classification==")
    score_2 = [s[1:]for s in SCORE_2]
    score_2 = np.mean(score_2, axis=0)
    print(f"Accuracy: {score_2[0]:0.4f}")
    print(f"F1: {score_2[1]:0.4f}")
    print(f"Recall: {score_2[2]:0.4f}")
    print(f"Precision: {score_2[3]:0.4f}")

    df_2 = pd.DataFrame(SCORE_2, columns=["Image", "Accuracy", "F1", "Recall", "Precision"])
    df_2.to_csv("files"+sep+domain+sep+"score_2_"+domain+"_"+name_model+"_"+str(dropout_rate)+".csv")
