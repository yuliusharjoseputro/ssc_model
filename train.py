import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import tensorflow_addons as tfa
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, AUC, MeanIoU, BinaryAccuracy
from config import get_config
import matplotlib.pyplot as plt
from model import UNet_Cblock, UNet_Res_Cblock, UNet_Res_SE_block, UNet_Res_ASPP_block, SSC
from metrics import dice_loss, logcoshDice, dice_coef, iou, focal_loss_non_weight
import time
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tensorflow.keras.utils import to_categorical
from collections import Counter
from scipy.stats import ttest_rel
from sklearn.model_selection import KFold
import csv
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from PIL import Image

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

def shuffling(x, y1, y2):
    x, y1, y2 = shuffle(x, y1, y2, random_state=42)
    return x, y1, y2

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
    x = x/255.0 #for mask denom
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
    
# Define the combined loss function
def combined_loss(y_true_seg, y_pred_seg, y_true_cls, y_pred_cls, alpha=0.5, beta=0.5):
    seg_loss = tf.keras.losses.binary_crossentropy(y_true_seg, y_pred_seg)
    cls_loss = tf.keras.losses.categorical_crossentropy(y_true_cls, y_pred_cls)
    return alpha * seg_loss + beta * cls_loss
    
def getCallBack(cfg):
    loss_param = {'segmentation_output_hybrid': logcoshDice, 'model_classification': focal_loss_non_weight(alpha=cfg["alpha_focal_loss"], gamma=cfg["gamma_focal_loss"])}
    metrics_param = {'segmentation_output_hybrid': [dice_coef, iou, MeanIoU(num_classes=2), BinaryAccuracy(), AUC(name='AUC_Seg'), Recall(name='Rec_Seg'), Precision(name='Prec_Seg')], 'model_classification': ['accuracy', AUC(name='AUC_Cls'), Recall(name='Rec_Cls'), Precision(name='Prec_Cls')]}
    
    monitor_modelcheckpoint_param = 'val_segmentation_output_hybrid_loss'
    monitor_reducelropplateau_param = 'val_segmentation_output_hybrid_loss'
    monitor_earlystopping_param = 'val_segmentation_output_hybrid_loss'
    
    return loss_param, metrics_param, monitor_modelcheckpoint_param, monitor_reducelropplateau_param, monitor_earlystopping_param

def train(train_ds, valid_ds, test_ds, cfg, dropout_rate, model_path, csv_path, history_path, model_name, n_fold, graph_acc_path, graph_loss_path, param_eval_acc, param_eval_loss, param_eval_val_acc, param_eval_val_loss, graph_acc_path_seg, graph_loss_path_seg, param_eval_acc_seg, param_eval_loss_seg, param_eval_val_acc_seg, param_eval_val_loss_seg):
    # combined_model_hybrid = UNet_Cblock(cfg, dropout_rate)
    # combined_model_hybrid = UNet_Res_Cblock(cfg, dropout_rate)
    # combined_model_hybrid = UNet_Res_SE_block(cfg, dropout_rate)
    # combined_model_hybrid = UNet_Res_ASPP_block(cfg, dropout_rate)
    combined_model_hybrid = SSC(cfg, dropout_rate)
    loss_param_hybrid, metrics_param_hybrid, monitor_modelcheckpoint_param_hybrid, monitor_reducelropplateau_param_hybrid, monitor_earlystopping_param_hybrid = getCallBack(cfg)
    combined_model_hybrid.compile(loss=loss_param_hybrid,
                            optimizer=Adam(cfg["lr"]),
                            metrics=metrics_param_hybrid
                            )
    callbacks_hybrid = [
        ModelCheckpoint(model_path, monitor=monitor_modelcheckpoint_param_hybrid, mode='min', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor=monitor_reducelropplateau_param_hybrid, factor=0.1, patience=cfg["patience_reduce_LRO"], min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor=monitor_earlystopping_param_hybrid,  mode='min', patience=cfg["patience_early_stopping"], restore_best_weights=False),
    ]

    start_hybrid = time.time()

    # Store initial weights
    init_weights = combined_model_hybrid.get_weights()

    history_hybrid = combined_model_hybrid.fit(
        train_ds,
        epochs=cfg["num_epochs"],
        validation_data=valid_ds,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks_hybrid,
    )
    
    end_hybrid = time.time()

    print("==Evaluate Model==")
    # Evaluate the model on the validation data
    custom_objects = {'iou': iou, 'dice_coef': dice_coef, 'logcoshDice':  logcoshDice, 'focal_loss_fixed': focal_loss_non_weight(alpha=cfg["alpha_focal_loss"], gamma=cfg["gamma_focal_loss"])}
    best_model_hybrid = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    result_k = best_model_hybrid.evaluate(valid_ds, verbose=0, return_dict = True)
    k_val_acc = result_k[param_eval_acc]
    k_val_loss = result_k[param_eval_loss]

    k_val_acc_seg = result_k[param_eval_acc_seg]
    k_val_loss_seg = result_k[param_eval_loss_seg]
    
    print(model_name + " - fold -", n_fold, " - validation accuracy:", k_val_acc, " - validation loss:", k_val_loss, " - validation dice coef:", k_val_acc_seg, " - validation dice loss:", k_val_loss_seg)

    train_acc = history_hybrid.history[param_eval_acc]
    val_acc = history_hybrid.history[param_eval_val_acc]

    train_loss = history_hybrid.history[param_eval_loss]
    val_loss = history_hybrid.history[param_eval_val_loss]

    train_acc_seg = history_hybrid.history[param_eval_acc_seg]
    val_acc_seg = history_hybrid.history[param_eval_val_acc_seg]

    train_loss_seg = history_hybrid.history[param_eval_loss_seg]
    val_loss_seg = history_hybrid.history[param_eval_val_loss_seg]

    print("==Generate Graphic==")
    generateGraphicLearningCurve(train_acc, val_acc, train_loss, val_loss, graph_acc_path, graph_loss_path)
    generateGraphicLearningCurve_Seg(train_acc_seg, val_acc_seg, train_loss_seg, val_loss_seg, graph_acc_path_seg, graph_loss_path_seg)

    # Save the training history to a file
    with open(history_path, 'wb') as file:
        pickle.dump(history_hybrid.history, file)

    
    combined_model_hybrid.set_weights(init_weights)
    
    return combined_model_hybrid, start_hybrid, end_hybrid, history_hybrid, k_val_acc, k_val_loss, k_val_acc_seg, k_val_loss_seg

def generateGraphicLearningCurve(train_acc, val_acc, train_loss, val_loss, graph_acc_path, graph_loss_path):

    # Create a plot of the learning curve
    plt.figure(figsize=(8, 8))
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Save the plot to a file
    plt.savefig(graph_acc_path)

    plt.figure(figsize=(8, 8))
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save the plot to a file
    plt.savefig(graph_loss_path)

def generateGraphicLearningCurve_Seg(train_acc, val_acc, train_loss, val_loss, graph_acc_path, graph_loss_path):

    # Create a plot of the learning curve
    plt.figure(figsize=(8, 8))
    plt.plot(train_acc, label='Training Dice Coef')
    plt.plot(val_acc, label='Validation Dice Coef')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Dice Coef')
    plt.legend()

    # Save the plot to a file
    plt.savefig(graph_acc_path)

    plt.figure(figsize=(8, 8))
    plt.plot(train_loss, label='Training Dice Loss')
    plt.plot(val_loss, label='Validation Dice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Dice Loss')
    plt.legend()

    # Save the plot to a file
    plt.savefig(graph_loss_path)
    
def generateAllGraphicAllFoldperModel_Seg(cfg, combined_history, graph_acc_path_all_fold, graph_loss_path_all_fold):
    # Retrieve the number of folds from the length of the history keys
    # k = len([key for key in combined_history_model_a.keys() if 'fold' in key])
    plt.figure(figsize=(8, 8))
    for i in range(1, cfg["num_folds"]+1):
        train_loss_key = f'fold_{i}_segmentation_output_hybrid_loss'
        val_loss_key = f'fold_{i}_val_segmentation_output_hybrid_loss'

        train_loss = combined_history[train_loss_key]
        val_loss = combined_history[val_loss_key]
        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, label=f'Fold {i} Train Dice Loss')
        plt.plot(epochs, val_loss, label=f'Fold {i} Val Dice Loss')

    plt.title('Training and Validation Dice Loss by Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.legend()

    plt.savefig(graph_loss_path_all_fold)

    plt.figure(figsize=(8, 8))
    for i in range(1, cfg["num_folds"]+1):
        train_acc_key = f'fold_{i}_segmentation_output_hybrid_dice_coef'
        val_acc_key = f'fold_{i}_val_segmentation_output_hybrid_dice_coef'

        train_acc = combined_history[train_acc_key]
        val_acc = combined_history[val_acc_key]
        epochs = range(1, len(train_acc) + 1)

        plt.plot(epochs, train_acc, label=f'Fold {i} Train Dice Coef')
        plt.plot(epochs, val_acc, label=f'Fold {i} Val Dice Coef')

    plt.title('Training and Validation Dice Coef by Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coef')
    plt.legend()

    #     Save the plot to a file
    plt.savefig(graph_acc_path_all_fold)

def generateAllGraphicAllFoldperModel(cfg, combined_history, graph_acc_path_all_fold, graph_loss_path_all_fold):
    # Retrieve the number of folds from the length of the history keys
    # k = len([key for key in combined_history_model_a.keys() if 'fold' in key])
    plt.figure(figsize=(8, 8))
    for i in range(1, cfg["num_folds"]+1):
        train_loss_key = f'fold_{i}_model_classification_loss'
        val_loss_key = f'fold_{i}_val_model_classification_loss'

        train_loss = combined_history[train_loss_key]
        val_loss = combined_history[val_loss_key]
        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, label=f'Fold {i} Train Loss')
        plt.plot(epochs, val_loss, label=f'Fold {i} Val Loss')

    plt.title('Training and Validation Loss by Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(graph_loss_path_all_fold)

    plt.figure(figsize=(8, 8))
    for i in range(1, cfg["num_folds"]+1):
        train_acc_key = f'fold_{i}_model_classification_accuracy'
        val_acc_key = f'fold_{i}_val_model_classification_accuracy'

        train_acc = combined_history[train_acc_key]
        val_acc = combined_history[val_acc_key]
        epochs = range(1, len(train_acc) + 1)

        plt.plot(epochs, train_acc, label=f'Fold {i} Train Acc')
        plt.plot(epochs, val_acc, label=f'Fold {i} Val Acc')

    plt.title('Training and Validation Accuracy by Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    #     Save the plot to a file
    plt.savefig(graph_acc_path_all_fold)

def denormalize(image):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = image * std + mean
    image = tf.clip_by_value(image, 0, 1)
    image = image * 255
    return image

if __name__ == "__main__":

    """Load Config Hyperparameter"""
    cfg = get_config()

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")
    master_folder = "new_data_hybrid_(80_20)_20_2class_denom_"

    """ Hyperparameters """
    domain = "radiography" #busi

    dataset_path = master_dataset+master_folder+domain+"_fold_"+str(cfg["image_size"]) 
    print(domain)

    """ Model """
    for dropout_rate in cfg["dropout_rate_list"]:
        
        fold = 0
        val_accuracies_a, val_losses_a, val_accuracies_b, val_losses_b, val_accuracies_c, val_losses_c, val_accuracies_d, val_losses_d = [],[],[],[],[],[],[],[]
        val_accuracies_a_seg, val_losses_a_seg, val_accuracies_b_seg, val_losses_b_seg, val_accuracies_c_seg, val_losses_c_seg, val_accuracies_d_seg, val_losses_d_seg = [],[],[],[],[],[],[],[]
        history_all_fold_model_a=[]
        start_time_all_fold_a, end_time_all_fold_a, start_time_all_fold_b, end_time_all_fold_b, start_time_all_fold_c, end_time_all_fold_c, start_time_all_fold_d, end_time_all_fold_d = 0, 0, 0, 0, 0, 0, 0, 0

        with open(master_path+'files'+sep+domain+sep+'result_avg_all_fold_all_model_'+str(dropout_rate)+'.csv', mode='w') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['model','fold', 'val_acc', 'val_loss', 'val_dice_coef', 'val_dice_loss','start time', 'end time', 'fold', 'image_size','num_channels', 'inputs', 'input_shape', 'batch_size', 'lr', 'weight_decay', 'beta_1', 'beta_2', 'epsilon','num_epochs', 'dropout_rate_list', 'num_folds', 'num_classes_segmentation', 'gamma_focal_loss', 'alpha_focal_loss', 'patience_early_stopping', 'patience_reduce_LRO', 'num_classes', 'class_names'])

            for fold in range(0, cfg["num_folds"]):

                fold += 1

                train_path = os.path.join(dataset_path+sep+"fold_"+str(fold), "train")
                valid_path = os.path.join(dataset_path+sep+"fold_"+str(fold), "valid")
                test_path = os.path.join(dataset_path, "test")

                train_x, train_y1, train_y2 = load_data(train_path)
                train_x, train_y1, train_y2 = shuffling(train_x, train_y1, train_y2)
                valid_x, valid_y1, valid_y2 = load_data(valid_path)
                test_x, test_y1, test_y2 = load_data(test_path)

                print(len(train_x))
                print(len(valid_x))
                print(len(test_x))

                train_dataset = tf_dataset(train_x, train_y1, train_y2, batch=cfg["batch_size"])
                valid_dataset = tf_dataset(valid_x, valid_y1, valid_y2, batch=cfg["batch_size"])
                test_dataset = tf_dataset(test_x, test_y1, test_y2, batch=cfg["batch_size"])

                train_steps_per_epoch = int(np.ceil(len(train_x) / cfg["batch_size"]))
                val_steps_per_epoch = int(np.ceil(len(valid_x) / cfg["batch_size"]))

                print(train_steps_per_epoch)
                print(val_steps_per_epoch)

                train_ds_abc = train_dataset.map(lambda x, y, z,: (x, (y, z)))
                valid_ds_abc = valid_dataset.map(lambda x, y, z: (x, (y, z)))
                test_ds_abc = test_dataset.map(lambda x, y, z: (x, (y, z)))

                print(train_ds_abc)
                print(valid_ds_abc)

                """checking distribution images within class"""
                if domain == "radiography":
                    train_y2_labels = [((path.split(sep)[-1]).split("_")[0]) for path in train_x]
                elif domain == "busi":
                    train_y2_labels = [((path.split(sep)[-1]).split("_")[0]) for path in train_x]
                
                labels = np.unique(train_y2_labels)
                label_counts = Counter(train_y2_labels)
                inc = 0
                inc_count = 0
                for label in labels:
                    inc+=1
                    count = label_counts[label]
                    inc_count+=count
                    print(f"Class {label}: {count} images")
                print(f"Total Class: {inc} - Total Images: {inc_count}")

                """ Generate class weights """
                if domain == "radiography":
                    train_x_labels = [cfg["class_names"].index((path.split(sep)[-1]).split("_")[0]) for path in train_x]
                elif domain == "busi":
                    train_x_labels = [cfg["class_names"].index((path.split(sep)[-1]).split("_")[0]) for path in train_x]

                y_train = to_categorical(train_x_labels, num_classes=cfg["num_classes"])

                # Calculate class frequencies
                class_frequencies = np.sum(y_train, axis=0) / np.sum(y_train)

                # Calculate inverse class weights
                class_weights = 1 / (class_frequencies + 1e-7)

                # Normalize weights
                class_weights /= np.sum(class_weights)

                # #Way 1
                class_weights = compute_class_weight("balanced", classes = np.unique(train_x_labels), y = train_x_labels)
                class_weight_dict = dict(zip(np.unique(train_x_labels), class_weights))

                print(class_weight_dict)


                print(f"Training fold {fold}")
                
                model_path_a = os.path.join(master_path+"files", domain+sep+"model_"+domain+"_"+cfg["name_model_a"]+"_"+str(fold)+"_"+str(dropout_rate)+".h5")
                csv_path_a = os.path.join(master_path+"files", domain+sep+"data_"+domain+"_"+cfg["name_model_a"]+"_"+str(fold)+"_"+str(dropout_rate)+".csv")
                history_path_a = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+str(fold)+"_"+str(dropout_rate)+".pkl")
                graph_acc_path_a = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+str(fold)+"_learning_curve_acc"+"_"+str(dropout_rate)+".png")
                graph_loss_path_a = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+str(fold)+"_learning_curve_loss"+"_"+str(dropout_rate)+".png")
                graph_acc_path_all_fold_a = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+"all_fold"+"_learning_curve_acc"+"_"+str(dropout_rate)+".png")
                graph_loss_path_all_fold_a = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+"all_fold"+"_learning_curve_loss"+"_"+str(dropout_rate)+".png")
                history_path_all_fold_a = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+"all_fold"+"_"+str(dropout_rate)+".pkl")
                graph_acc_path_a_seg = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+str(fold)+"_learning_curve_acc_seg"+"_"+str(dropout_rate)+".png")
                graph_loss_path_a_seg = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+str(fold)+"_learning_curve_loss_seg"+"_"+str(dropout_rate)+".png")
                graph_acc_path_all_fold_a_seg = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+"all_fold"+"_learning_curve_acc_seg"+"_"+str(dropout_rate)+".png")
                graph_loss_path_all_fold_a_seg = os.path.join(master_path+"files", domain+sep+"history_"+domain+"_"+cfg["name_model_a"]+"_"+"all_fold"+"_learning_curve_loss_seg"+"_"+str(dropout_rate)+".png")

                # evaluate per fold
                param_eval_acc_abc = "model_classification_accuracy"
                param_eval_loss_abc = "model_classification_loss"
                param_eval_val_acc_abc = "val_model_classification_accuracy"
                param_eval_val_loss_abc = "val_model_classification_loss"

                param_eval_acc_abc_seg = "segmentation_output_hybrid_dice_coef"
                param_eval_loss_abc_seg = "segmentation_output_hybrid_loss"
                param_eval_val_acc_abc_seg = "val_segmentation_output_hybrid_dice_coef"
                param_eval_val_loss_abc_seg = "val_segmentation_output_hybrid_loss"

                combined_model_hybrid_a, start_hybrid_a, end_hybrid_a, history_hybrid_a, k_val_acc_a, k_val_loss_a, k_val_acc_a_seg, k_val_loss_a_seg, = train(train_ds_abc, valid_ds_abc, test_ds_abc, cfg, dropout_rate, model_path_a, csv_path_a, history_path_a, cfg["name_model_a"], fold, graph_acc_path_a, graph_loss_path_a, param_eval_acc_abc, param_eval_loss_abc, param_eval_val_acc_abc, param_eval_val_loss_abc, graph_acc_path_a_seg, graph_loss_path_a_seg, param_eval_acc_abc_seg, param_eval_loss_abc_seg, param_eval_val_acc_abc_seg, param_eval_val_loss_abc_seg)
            
                val_accuracies_a.append(k_val_acc_a)
                val_losses_a.append(k_val_loss_a)  

                val_accuracies_a_seg.append(k_val_acc_a_seg)
                val_losses_a_seg.append(k_val_loss_a_seg)  
                
                writer.writerow([cfg["name_model_a"], fold, k_val_acc_a, k_val_loss_a, k_val_acc_a_seg, k_val_loss_a_seg, start_hybrid_a, end_hybrid_a, fold, cfg['image_size'], cfg['num_channels'], cfg['inputs'], cfg['input_shape'], cfg['batch_size'], cfg['lr'], cfg['weight_decay'], cfg['beta_1'], cfg['beta_2'], cfg['epsilon'], cfg['num_epochs'], cfg['dropout_rate_list'], cfg['num_folds'], cfg['num_classes_segmentation'], cfg['gamma_focal_loss'], cfg['alpha_focal_loss'], cfg['patience_early_stopping'], cfg['patience_reduce_LRO'], cfg['num_classes'], cfg['class_names']])
                start_time_all_fold_a += start_hybrid_a
                end_time_all_fold_a += end_hybrid_a
                history_all_fold_model_a.append(history_hybrid_a)

            print("Training model needs %0.2f seconds to complete" % (end_time_all_fold_a - start_time_all_fold_a))

            avg_accuracy_a = sum(val_accuracies_a) / cfg["num_folds"]
            avg_loss_a = sum(val_losses_a) / cfg["num_folds"]
            print("Average validation accuracy model a:", avg_accuracy_a)
            print("Average validation loss model a:", avg_loss_a)

            avg_accuracy_a_seg = sum(val_accuracies_a_seg) / cfg["num_folds"]
            avg_loss_a_seg = sum(val_losses_a_seg) / cfg["num_folds"]
            print("Average validation dice coef model a:", avg_accuracy_a_seg)
            print("Average validation dice loss model a:", avg_loss_a_seg)
            
            combined_history_model_a = {}
            for i, fold_history in enumerate(history_all_fold_model_a):
                for metric in fold_history.history.keys():
                    key = f'fold_{i+1}_{metric}'
                    combined_history_model_a[key] = fold_history.history[metric]
                    
            with open(history_path_all_fold_a, 'wb') as f:
                pickle.dump(combined_history_model_a, f)
            
            generateAllGraphicAllFoldperModel(cfg, combined_history_model_a, graph_acc_path_all_fold_a, graph_loss_path_all_fold_a)
            generateAllGraphicAllFoldperModel_Seg(cfg, combined_history_model_a, graph_acc_path_all_fold_a_seg, graph_loss_path_all_fold_a_seg)