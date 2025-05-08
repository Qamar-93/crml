# testing the utility of the models by testing them against clean data only
from math import ceil
import os
import json
import numpy as np
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
from Training.ModelTrainer import ModelTrainer
from utils.training_utils import CustomLoss, CustomMetric
from utils.helpers import evaluate_and_plot, evaluate
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights
import torch
import tensorflow as tf
import pandas as pd
def main(res_folder, json_path, loss_fuction, noise_type, epsilon=0.5):
    physical_devices = tf.config.list_physical_devices('GPU')
    
    tf.config.set_visible_devices(physical_devices[1],'GPU')  # for using the first GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    res_folder = res_folder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #### reading the config file    
    with open(json_path) as f:
        configs = json.load(f)

    #### initialize the data generators
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    loss_function = loss_fuction
    
       
    for config in configs["models"]:
        model_type = config["type"] 
        training_type = config["training_type"]
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        plots_folder = f"{res_folder}/{config['type']}/{config['training_type']}/plots"   
        
        input_shape = num_inputs   
        ####### evaluate model robustness
        ########### create new data
        # x_len = 32
        x_len = configs["num_samples"]
        # x_len=2
        
        num_noises = 40
        variance = 0.3
        distribution = "normal"
        
        test_noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=distribution, variance=variance)

        test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)
        
        x_clean_orig, y_clean_orig = test_dataset_generator.generate_dataset()
        
        
        x_clean_orig, y_clean_orig = test_dataset_generator.meshgrid_x_y(x_clean_orig)
        num_input_features = x_clean_orig.shape[1]
        if num_input_features > 1:
            x_clean_orig = x_clean_orig.reshape(x_clean_orig.shape[0], -1).T
        y_clean_orig = y_clean_orig.ravel()


        test_dataset_generator.num_samples = x_clean_orig.shape[0]
        test_dataset_generator.noise_generator.num_samples = test_dataset_generator.num_samples
        
        
        # load models:       
        model_path = config["model_path"]
        models = []         
        models_num = 9
        
        for i in range(models_num - 1): # -1 because we removed the worst model
            model_path_i = f"{model_path}/model_{i+1}"
            print(model_path_i)
            print("input_shape", input_shape)
            trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
            model = trainer.model
                
            if loss_function == "custom_loss":
                customloss = CustomLoss(model=model, metric=None, y_clean=y_clean_orig, x_noisy=x_clean_orig, len_input_features=input_shape, bl_ratio=0)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                # import tensorflow_privacy        
                # optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
                #     l2_norm_clip=0.7,
                #     noise_multiplier=2.1,
                #     num_microbatches=1,
                #     learning_rate=0.001,
                # )
                # optimizer="adam"
                model.compile(optimizer=optimizer, loss=customloss)
                # model.load_weights(f"{model_path_i}/model_weights.h5")
                # check if model weights are loaded correctly by printing the weights before and after loading
                print("model weights before loading", model.get_weights())
                trainer.load_model(model_obj=model, filepath=f"{model_path_i}")
                print("model weights after loading", model.get_weights())
                print("model loaded with custom loss")
                
            else:
                if config["type"] == "linear":
                    # optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
                    # l2_norm_clip=0.7,
                    # noise_multiplier=2.1,
                    # num_microbatches=1,
                    # learning_rate=0.001,
                    # )
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                    model.compile(optimizer=optimizer, loss=loss_function)
                    # model.load_weights(f"{model_path_i}/model_weights.h5")  
                    trainer.load_model(model_obj=model, filepath=f"{model_path_i}")  
                else:
                    model = trainer.load_model(model_obj=model, filepath=f"{model_path_i}")
                print("model loaded with mse")
            models.append(model)
            
        print("len of models", len(models))
        # random_seeds_all should be of length = no_noisy_tests, where each element is a set of random seeds for each input feature
        no_noisy_tests = len(models)
        
        y_pred_all = {j : None for j in range(1, no_noisy_tests)}
        for idx, model in enumerate(models):
            # if config["load"] == True:
            #     model_path_i = f"{model_path}/model_{idx+1}"
            # else:
            #     model_path_i = f"{models_folder}/model_{idx+1}"
            # if not os.path.exists(model_path_i):
            #     os.makedirs(model_path_i)    
            
            # model_i_res_folder = f"{models_folder}/rm_results_clean_data/model_{idx+1}"
            # #### new3 is new2 again but for 5 models only
            # if not os.path.exists(model_i_res_folder):
            #     os.makedirs(model_i_res_folder)            
            
            # for i in range(no_noisy_tests):
            y_pred_all[idx+1] = model.predict(x_clean_orig).ravel()
            
        print("y_pred_all", y_pred_all)
        y_pred_all = np.array(list(y_pred_all.values()))
        y_pred_all_min = np.min(y_pred_all, axis=0)
        y_pred_all_max = np.max(y_pred_all, axis=0)
        print("is min and max the same", np.all(y_pred_all_min == y_pred_all_max))
        import matplotlib.pyplot as plt
        plt.clf()
        # for i in range(no_noisy_tests):
        #     plt.plot(np.linspace(0, len(y_pred_all[i]), len(y_pred_all[i])), y_pred_all[i])
        # plt.fill_between(y_clean_orig, y_pred_all_min, y_pred_all_max, alpha=0.5, color = 'red', label="Prediction bounds")
        plt.plot(y_clean_orig, y_clean_orig, label="True")
        plt.plot(y_clean_orig, y_pred_all[0], label="Predicted")
        
        plt.legend()
        # save the data as txt file
        
        df = pd.DataFrame(data={"y_true": y_clean_orig, "y_pred": y_pred_all[0],
                                "x": x_clean_orig.ravel()})
        df.to_csv(f"{res_folder}/predicted_vs_true_{model_type}_{loss_fuction}_{noise_type}.txt")
        plt.savefig(f"./predicted_vs_true_{model_type}_{loss_fuction}_{noise_type}.png")
if __name__ == '__main__':
    json_files = [f for f in os.listdir('./configs/equations_all/') if f.endswith('.json')]
    json_files = [
        # "I_6_2.json",
        "I_6_2a.json",
        # "I_6_2b.json",
        ]
    # Iterate over the JSON files
    for json_file in json_files:
        # Set the res_folder variable based on the name of the JSON file
        # res_folder = f"./results_mse_dp_{os.path.splitext(json_file)[0]}"
        # noise_type = "laplace"
        noise_type = "normal"
        if noise_type == "normal":
            # # loss_fuction = "mse"
            # res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{loss_fuction}/{noise_type}/non-dp/"
            json_path = f'./configs/equations_all/{json_file}'

            # main(res_folder, json_path, loss_fuction="mse", noise_type=noise_type)

            loss_fuction = "custom_loss"
            res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{loss_fuction}/{noise_type}/non-dp/"
            main(res_folder, json_path, loss_fuction="custom_loss", noise_type=noise_type)
        elif noise_type == "laplace_dp" or noise_type == "laplace":
            epsilon = [0.1, 1, 10]
            
            for eps in epsilon:
                loss_fuction = "mse"
                res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{loss_fuction}/{noise_type}/epsilon_{eps}/"

                json_path = f'./configs/equations_all/{json_file}'
                main(res_folder, json_path, loss_fuction="mse", noise_type=noise_type, epsilon=eps)
                loss_fuction = "custom_loss"
                res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{loss_fuction}/{noise_type}/epsilon_{eps}/"
                main(res_folder, json_path, loss_fuction="custom_loss", noise_type=noise_type, epsilon=eps)