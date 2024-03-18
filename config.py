from tensorflow.keras.layers import Input
def get_config():

    """ Hyperparameters """
    hp = {}
    hp["image_size"] = 256 #64
    hp["num_channels"] = 3
    hp["inputs"] = Input((hp["image_size"], hp["image_size"], hp["num_channels"]))
    hp["input_shape"] = (hp["image_size"], hp["image_size"], hp["num_channels"])

    hp["batch_size"] = 8 #16
    hp["lr"] = 1e-4 #1e-4
    hp["num_epochs"] = 120
    hp["dropout_rate_list"] = [0.3]##[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    hp["num_folds"] = 5
    hp["num_classes_segmentation"] = 1
    hp["gamma_focal_loss"] = 1
    hp["alpha_focal_loss"] = 0.1
    hp["patience_early_stopping"] = 30 #30
    hp["patience_reduce_LRO"] = 10 #10

    #for t-test
    # hp["name_model_a"] = "UNet_Cblock"
    # hp["name_model_a"] = "UNet_Res_Cblock"
    # hp["name_model_a"] = "UNet_Res_SE_block"
    # hp["name_model_a"] = "UNet_Res_APP_block"
    hp["name_model_a"] = "SSC"
 

    hp["num_classes"] = 4
    hp["class_names"] = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]

    # hp["num_classes"] = 2
    # hp["class_names"] = ["benign", "malignant"]

    return hp
