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
from utils.dists import L2_distance
import datetime
# def main(res_folder, loss_function):
def main(res_folder, json_path, loss_fuction, noise_type, epsilon=0.5):
    physical_devices = tf.config.list_physical_devices('GPU')
    
    tf.config.set_visible_devices(physical_devices[1],'GPU')  # for using the first GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # x_len = 7
    num_noises = 20
    # num_noises = 2
    distribution = noise_type
    variance = 0.3
    
    res_folder = res_folder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #### reading the config file    
    with open(json_path) as f:
        configs = json.load(f)

    x_len = configs["num_samples"]

    #### initialize the data generators
    # noise_model = NoiseGenerator(x_len, num_noises, distribution, variance)
    noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=distribution, variance=variance, epsilon=epsilon)
    equation_str = configs["equation"]
    input_features = configs["features"]
    num_inputs = len(input_features)
    
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    metric = RobustnessMetric()
    input_shape = num_inputs
    # loss_function = 'custom_loss'
    loss_function = loss_fuction
       
    for config in configs["models"]:
        training_type = config["training_type"]
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        plots_folder = f"{res_folder}/{config['type']}/{config['training_type']}/plots"   
            
        xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(config, metric_instance=metric)
        

        x_noisy = xy_noisy[0]
        y_noisy = xy_noisy[1]

        x_clean, y_clean = xy_clean
        
        if training_type == "noise-aware":
            # x_noisy = np.repeat(xy_noisy[0], 2, axis=2)
            x_noisy = np.tile(xy_noisy[0], (1, 1, 2))
            # x_noisy = np.tile(xy_noisy[0], (1, 1, 3))
            # instead set the last n columns to 0, where n is the number of noise features
            x_noisy[:, :, num_inputs:] = 0
            x_noisy[:, :, num_inputs:2*num_inputs] = gx_gy[0]
            # now the third n columns are x_noisy - x_clean
            # x_noisy[:, :, 2*num_inputs:] = gx_gy[0] - xy_clean[0]
            input_shape = num_inputs * 2
            # input_shape = num_inputs * 3
            xy_noisy = (x_noisy, y_noisy)
    
        x_noisy, y_noisy = xy_noisy
        x_clean, y_clean = xy_clean
        
        gx, gx_y = gx_gy
       
        indices_train, indices_valid = indices
        
       
        x_noisy_train = x_noisy[:, indices_train, :]
        x_noisy_valid = x_noisy[:, indices_valid, :]
        
        y_noisy_train = y_noisy[:, indices_train, ]
        y_noisy_valid = y_noisy[:, indices_valid, ]

        x_clean_train = x_clean[indices_train, :]
        x_clean_valid = x_clean[indices_valid, :]
        y_clean_train = y_clean[indices_train,]
        y_clean_valid = y_clean[indices_valid,]
        print("mse of train y and noisy y", np.mean((y_clean_train - y_noisy_train)**2))
        
        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
        print("config['type']", config["type"])
        ############## calculate baseline metric for the equation itself for both training and validation sets
        print(y_clean_train.shape, y_noisy_train.shape, y_clean_valid.shape, y_noisy_valid.shape)

        y_noisy_bl_train = np.zeros((y_noisy_train.shape[0], y_noisy_train.shape[1]))
        y_noisy_bl_valid = np.zeros((y_noisy_valid.shape[0], y_noisy_valid.shape[1]))

        # if x_clean_train.shape[1] > num_inputs: # more features are added, so we use the original features only
        for idx_shape, x_noise_vector in enumerate(x_noisy_train):
            
            y_noise_vector = dataset_generator.apply_equation(x_noise_vector[:, :num_inputs])
            y_noisy_bl_train[idx_shape, :] = y_noise_vector.flatten()
        
        outer_dists = ["Euclidean"]
        
        ############################# calculate the nominator of the metric in case of custom loss
        gxs_dists = []
        for i in range(num_inputs):
            gx = metric.extract_g(x_clean_valid[:, i], x_hat=x_noisy_valid[:, :, i])
            gx, x_clean_valid_i_scaled = metric.rescale_vector(true=x_clean_valid[:, i], noisy=gx)
            gxs_dists.append(L2_distance(gx, x_clean_valid_i_scaled, type="overall"))

        ############################# calculate the correct weights of the expression as a function of the input features, using Salib
        correct_weights = estimate_weights(model_path=f"{res_folder}/baseline/", inputs=input_features, dataset_generator=dataset_generator,
                                           num_samples=1000, model_type="expression") 
        
        if training_type == "noise-aware":
            correct_weights = np.append(correct_weights, np.zeros(len(correct_weights)))
            gxs_dists = np.append(gxs_dists, np.zeros(len(gxs_dists)))
        
        # bl_gy = metric.extract_g(y_clean_train, x_hat=y_noisy_bl_train)
        
        # bl_gy, y_clean_train_scaled = metric.rescale_vector(true=y_clean_train, noisy=bl_gy)
  
        # for i in range(x_noisy_train.shape[2]):
        #     max_x = np.max(x_noisy_train[:, :, i], axis=0)
        #     min_x = np.min(x_noisy_train[:, :, i], axis=0)
        #     plt.clf()
        #     plt.plot(np.linspace(0, x_noisy_train.shape[1], x_noisy_train.shape[1]), max_x, label="max_x")
        #     plt.plot(np.linspace(0, x_noisy_train.shape[1], x_noisy_train.shape[1]), min_x, label="min_x")
        #     plt.plot(np.linspace(0, x_noisy_train.shape[1], x_noisy_train.shape[1]), x_clean_train[:, i], label="clean")
        #     plt.legend()
        #     plt.savefig(f"./noisy_clean_{i}.png")
                
        # print("clean", y_clean_train)
        # print("noisy", y_noisy_bl_train)
        # max_noisy = np.max(y_noisy_bl_train, axis=0)
        # min_noisy = np.min(y_noisy_bl_train, axis=0)
        # plt.clf()
        # plt.plot(np.linspace(0, len(max_noisy), len(max_noisy)), max_noisy, label="max_noisy")
        # plt.plot(np.linspace(0, len(min_noisy), len(min_noisy)), min_noisy, label="min_noisy")
        # # plt.plot(np.linspace(0, len(y_clean_train), len(y_clean_train)), y_clean_train, label="clean")
        # plt.legend()
        # plt.savefig(f"./noisy_clean.png")
        print("correct_weights", correct_weights, "x_clean_train", x_clean_train.shape, "y_clean_train", y_clean_train.shape, "x_noisy_train", x_noisy_train.shape, "y_noisy_bl_train", y_noisy_bl_train.shape)
        rm_bl_train = metric.calculate_metric(x_clean_train, y_clean_train, 
                                              x_hat=x_noisy_train, y_hat=y_noisy_bl_train,
                                              outer_dist=outer_dists, weights=correct_weights, path=f"{res_folder}/baseline/training")["Euclidean"]
        
        print("rm_bl_train", rm_bl_train["Input distance"], rm_bl_train["Output distance"], rm_bl_train["Ratio"])
        with open(f"{res_folder}/baseline/training/rm.txt", "w") as outfile:
            outfile.write(str(rm_bl_train))
        # for idx_shape, x_noise_vector in enumerate(x_noisy_valid):
        #     y_noise_vector = dataset_generator.apply_equation(x_noise_vector)
        #     y_noisy_bl_valid[idx_shape, :] = y_noise_vector.flatten()
        # rm_bl_valid = metric.calculate_metric(x_clean_valid, y_clean_valid,
        #                                         x_hat=x_noisy_valid, y_hat=y_noisy_bl_valid,
        #                                         outer_dist=outer_dists, weights=correct_weights, path=f"{res_folder}/baseline/validation")["Euclidean"]["Ratio"]
        # with open(f"{res_folder}/baseline/validation/rm.txt", "w") as outfile:
        #     outfile.write(str(rm_bl_valid))
        rm_bl_train = rm_bl_train["Output distance"]
        # rm_bl_train = 2.9
        # print("rm_bl_train", rm_bl_train)
        
        # multiply each gx by the corresponding weight
        gxs_dists = gxs_dists * correct_weights
        gxs_dists = np.sum(gxs_dists)

        ####################################################
        if config["training_type"] == "clean":
            models_num = 10
        else:
            models_num = 10
                   
        if config["load"] == True:
            model_path = config["model_path"]
            model_path = model_path.replace("epsilon_0.1", f"epsilon_{epsilon}")
            model_path = model_path.replace("loss_custom_loss", f"loss_{loss_function}")
            if "models_all" in model_path:
                models = []
                # find the latest time stamp from the saved ones
                losses_time_stamps = [f for f in os.listdir(model_path) if f.startswith("losses")]
                latest_time_stamp = sorted(losses_time_stamps)[-1]
                losses = np.loadtxt(f"{model_path}/{latest_time_stamp}")
                    
                # for i in range(models_num - 1): # -1 because we removed the worst model
                for i in range(models_num): # -1 because we removed the worst model
                    model_path_i = f"{model_path}/model_{i+1}"
                    print(model_path_i)
                    # if input_shape != x_clean.shape[1]:
                    #     input_shape = x_clean.shape[1]
                    print("input_shape", input_shape)
                    trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
                    model = trainer.model
                        
                    if loss_function == "custom_loss":
                        # model = keras.models.load_model(f"{model_path_i}/model.pkl", compile=False)
                        customloss = CustomLoss(model=model, metric=metric, y_clean=y_clean_train, x_noisy=x_noisy_train, len_input_features=input_shape, bl_ratio=rm_bl_train)
                        # trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
                        # model = trainer.model
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
                        # optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
                        # l2_norm_clip=0.7,
                        # noise_multiplier=2.1,
                        # num_microbatches=1,
                        # learning_rate=0.001,
                        # )
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                        model.compile(optimizer=optimizer, loss=loss_function)
                        # model.load_weights(f"{model_path_i}/model_weights.h5")  
                        print("model weights before loading", model.get_weights())
                        trainer.load_model(model_obj=model, filepath=f"{model_path_i}")  
                        print("model loaded with mse")
                        print("model weights after loading", model.get_weights())
                        # model = trainer.load_model(f"{model_path_i}/model.pkl")
                        # calculate the mse on the training data
                        training_mse = np.mean((xy_train[1] - model.predict(xy_train[0]))**2)
                        print("training_mse", training_mse)
                        print("mse on validation data", np.mean((xy_valid[1] - model.predict(xy_valid[0]))**2))
                        print("mse on test data", np.mean((xy_test[1] - model.predict(xy_test[0]))**2))
                        
                    models.append(model)
                print("len of models", len(models))
                print("losses", losses)
                best_model = models[np.argmin(losses)]
            else:
                
                best_model = trainer.load_model(f"{model_path}/model.pkl")
                best_model.to(device)
                models = [best_model]
            
            history = None
            best_history = None
            plot_path = f"{model_path}/plot"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

        else:
            best_model = None
            best_model_path = None
            best_history = None
            models = []
            losses = []
            valid_losses = []
            valid_losses_all_epochs = []
            last_epoch = []
            rm_vals = []
            if loss_function == "custom_loss":
                config["fit_args"]["metric"] = metric
                # config["fit_args"]["x_noisy"] = tf.convert_to_tensor(xy_noisy[0], dtype=tf.float64)
                config["fit_args"]["x_noisy_valid"] = tf.convert_to_tensor(x_noisy_valid, dtype=tf.float64)
                config["fit_args"]["x_noisy_train"] = tf.convert_to_tensor(x_noisy_train, dtype=tf.float64)
                config["fit_args"]["len_input_features"] = input_shape
                config["fit_args"]["bl_ratio"] = tf.convert_to_tensor(rm_bl_train, dtype=tf.float64)
                config["fit_args"]["nominator"] = tf.convert_to_tensor(gxs_dists, dtype=tf.float64)
                config["fit_args"]["y_clean_valid"] = tf.convert_to_tensor(y_clean_valid, dtype=tf.float64)
                config["fit_args"]["y_clean_train"] = tf.convert_to_tensor(y_clean_train, dtype=tf.float64)
            
            for i in range(models_num):
                print("models folder", models_folder)
                if not os.path.exists(models_folder):
                    os.makedirs(models_folder)
                else:
                    if os.path.exists(f"{models_folder}/model_{i+1}"):
                        print(f"model_{i+1} already exists")
                        continue
                print("model", i+1)
                model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
                
                if best_model is None or history.history['val_loss'][-1] < best_history.history['val_loss'][-1]:
                    best_model = model
                    best_history = history
                    best_model_path = f"{models_folder}/model_{i+1}"
                            
                models.append(model)
                if history is not None:
                    if loss_function == "custom_loss":
                        losses.append(history.history['mse'][-1])
                        valid_losses.append(history.history['val_mse'][-1])
                        rm_vals.append(history.history['custom_metric'])
                        valid_losses_all_epochs.append(history.history['val_mse'])
                    else:
                        losses.append(history.history['loss'][-1])
                        valid_losses.append(history.history['val_loss'][-1])
                        valid_losses_all_epochs.append(history.history['val_loss'])
                    last_epoch.append(len(history.history['loss']))
                    
                else:
                    losses.append(np.inf)
                    valid_losses.append(np.inf)                    
            # if models_num > 1:
            #     # execlude the worst model out of the 11 models
            #     worst_model = models[np.argmax(losses)]
            #     models.remove(worst_model)
            #     losses.remove(np.max(losses))
            #     valid_losses.remove(np.max(valid_losses))
                
            # save the other 10 models in a folder called models_all, each with name model_1, model_2, etc.
            models_all_path = f"{models_folder}/{config['model_path']}"
            if not os.path.exists(models_all_path):
                os.makedirs(models_all_path)
                
            for i, model in enumerate(models):
                if not os.path.exists(f"{models_folder}/model_{i+1}"):
                    os.makedirs(f"{models_folder}/model_{i+1}")
                # model.save_weights(f"{models_folder}/model_{i+1}/model_weights.h5")
                trainer.save_model(model, f"{models_folder}/model_{i+1}")

                # trainer.model = model
                # trainer.save_model(f"{models_folder}/model_{i+1}")           
            # save the losses list in a txt file, with time stamp
            time_stamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
            with open(f"{models_folder}/losses_{time_stamp}.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in losses))
            with open(f"{models_folder}/valid_losses_{time_stamp}.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in valid_losses))
            with open(f"{models_folder}/last_epoch_{time_stamp}.txt", "w") as outfile:
                outfile.write(str(last_epoch))
            with open(f"{models_folder}/rm_vals_{time_stamp}.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in rm_vals))
            with open(f"{models_folder}/valid_losses_all_epochs_{time_stamp}.txt", "w") as outfile:
                outfile.write("\n".join(str(item) for item in valid_losses_all_epochs))
            # trainer.model = best_model
            # trainer.save_model(f"{model_path}")     
            
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)
        trainer.model = best_model
        if best_model is not None:
            print("best model", best_model)
            # evaluate_and_plot(trainer, best_history, xy_test, f"{plots_folder}")

####### evaluate model robustness
        ########### create new data
        # x_len = 32
        x_len = configs["num_samples"]
        # x_len = 100
        num_noises = 40
        variance = 0.3
        # if the random seed is 0, then it will be randomly generated, otherwise it will be as specified
        
        # test_noise_model = NoiseGenerator(x_len, num_noises, distribution, variance)
        print("epsilon is ")
        test_noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=distribution, variance=variance, epsilon=epsilon)

        x_noisy_new = None
        test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)
        x_clean_orig, y_clean_orig = test_dataset_generator.generate_dataset()
        
        # # apply log transformation to the clean data
        # x_clean = np.log(x_clean)
        # y_clean = np.log(y_clean)
        
        # apply the meshgrid to x and y clean
        y_clean_orig = y_clean_orig.ravel()
        
        x_clean_orig, y_clean_orig = test_dataset_generator.meshgrid_x_y(x_clean_orig)
        num_input_features = x_clean_orig.shape[1]
        if num_input_features > 1:
            x_clean_orig = x_clean_orig.reshape(x_clean_orig.shape[0], -1).T
        y_clean_orig = y_clean_orig.ravel()

        original_num_samples = test_dataset_generator.num_samples
        test_dataset_generator.num_samples = x_clean_orig.shape[0]
        test_dataset_generator.noise_generator.num_samples = test_dataset_generator.num_samples
        
        
        # create ten different noisy sets
        outer_dists = ["Euclidean", "L1"]
        # random_seeds_all should be of length = no_noisy_tests, where each element is a set of random seeds for each input feature
        no_noisy_tests = 1

        random_seeds_all = [np.linspace(0, 1000, num_inputs, dtype=int) for _ in range(no_noisy_tests)]
        
        for idx, model in enumerate(models):
            print("Number of models is", len(models))
            print("model", idx)

            if config["load"] == True:
                model_path_i = f"{model_path}/model_{idx+1}"
                
            else:
                model_path_i = f"{models_folder}/model_{idx+1}"
            if not os.path.exists(model_path_i):
                os.makedirs(model_path_i)    
            
            model_i_res_folder = f"{models_folder}/rm_results_try/model_{idx+1}"
            #### new3 is new2 again but for 5 models only
            if not os.path.exists(model_i_res_folder):
                os.makedirs(model_i_res_folder)            
            target_feats_ids = [el for el in range(len(input_features))]
            rm_worst_output = None
            # for target_feat_idx in target_feats_ids:
            x_noisy_all = {j: None for j in range(no_noisy_tests)}
            y_noisy_all = {k: None for k in range(no_noisy_tests)}
            weights_all = {l: None for l in range(no_noisy_tests)}
            rms = {rm_i: None for rm_i in range(no_noisy_tests)}
            
            for i_noisy in range(no_noisy_tests):
                print("i_noisy", i_noisy, "model", idx, "x_clean_orig", x_clean_orig.shape, "y_clean_orig", y_clean_orig.shape)
                x_noisy, y_noisy = test_dataset_generator.modulate_clean(x_clean_orig, y_clean_orig, target_feat_idx=target_feats_ids, random_seeds=random_seeds_all[i_noisy])
                
                x_noisy_all[i_noisy] = x_noisy
                y_noisy_all[i_noisy] = y_noisy
            
            for key_rm, value in x_noisy_all.items():
                x_noisy = value
                y_noisy = y_noisy_all[key_rm]
                
                ########### estimate the weights of the input features
                y_noisy_new = np.zeros((y_noisy.shape[0], y_noisy.shape[1]))
                
                if training_type == "noise-aware":
                    # extract gx from each x feature in x_noisy and x
                    # gx = np.zeros((x_len, test_dataset_generator.num_inputs))
                    gx = np.zeros((test_dataset_generator.num_samples, test_dataset_generator.num_inputs))
                    gx_y = np.zeros((test_dataset_generator.num_samples,))
                    for idx_shape in range(x_noisy.shape[2]):
                        gx_temp = metric.extract_g(x_clean_orig[:, idx_shape], x_hat=x_noisy[:, :, idx_shape])
                        gx[:, idx_shape] = gx_temp
                                                
                gx_y = y_clean_orig
                
                sampling_rate = ceil(x_clean_orig.shape[0]/((original_num_samples*2)*(x_clean_orig.shape[1])))
                print(f"Downsampling the data to {sampling_rate} of the original size")

                x_clean_orig = x_clean_orig[::sampling_rate]
                y_clean_orig = y_clean_orig[::sampling_rate]
                x_noisy = x_noisy[:, ::sampling_rate]
                y_noisy = y_noisy[:, ::sampling_rate]
                gx = gx[::sampling_rate]
                gx_y = gx_y[::sampling_rate]
                
                # baseline metric test
                y_noisy_bl_test = np.zeros((y_noisy.shape[0], y_noisy.shape[1]))
                
                # if x_clean_train.shape[1] > num_inputs: # more features are added, so we use the original features only
                for idx_shape, x_noise_vector in enumerate(x_noisy):
                    
                    y_noise_vector = dataset_generator.apply_equation(x_noise_vector[:, :num_inputs])
                    y_noisy_bl_test[idx_shape, :] = y_noise_vector.flatten()
                
                outer_dists = ["Euclidean"]
                print("x_clean_orig", x_clean_orig.shape, "y_clean_orig", y_clean_orig.shape, "x_noisy", x_noisy.shape, "y_noisy", y_noisy.shape)
                rm_bl_test = metric.calculate_metric(x_clean_orig, y_clean_orig, 
                                                      x_hat=x_noisy, y_hat=y_noisy_bl_test,
                                                      outer_dist=outer_dists, weights=correct_weights, path=f"{res_folder}/baseline/testing_laplace")["Euclidean"]
                
                print("rm_bl_test", rm_bl_test["Input distance"], rm_bl_test["Output distance"], rm_bl_test["Ratio"])
                with open(f"{res_folder}/baseline/testing_laplace/rm.txt", "w") as outfile:
                    outfile.write(str(rm_bl_test))

                # for noise-aware training, we need to add the gx to the x_noisy
                if training_type == "noise-aware":
                    x_noisy_new = np.zeros((x_noisy.shape[0], x_noisy.shape[1], x_noisy.shape[2] * 2))
                    for idx_shape in range(x_noisy.shape[2]):
                        # repeat the noisy feature
                        # x_noisy_new[:, :, idx_shape] = x_noisy[:, :, idx_shape]
                        # x_noisy_new[:, :, idx_shape + x_noisy.shape[2]] = x_noisy[:, :, idx_shape]
                        x_noisy_new[:, :, idx_shape] = x_noisy[:, :, idx_shape]
                        x_noisy_new[:, :, idx_shape + x_noisy.shape[2]] = gx[:, idx_shape]

                else:
                    x_noisy_new = x_noisy
                    
                y_noisy_new = y_noisy
                
                #### calculate the weights of the input features
                if training_type == "noise-aware":
                    if loss_function == "custom_loss":
                        weights = estimate_weights(f"{model_path_i}", input_features, test_dataset_generator, training_type="noise-aware", num_samples=x_len, loss_function=loss_function, metric=metric, 
                                                   x_noisy=x_noisy_train, len_input_features=input_shape, bl_ratio=rm_bl_train, y_clean=y_clean_train, model_type=config["type"])
                    else:
                        weights = estimate_weights(f"{model_path_i}", input_features, test_dataset_generator, training_type="noise-aware", num_samples=x_len, loss_function=loss_function, model_type=config["type"])
                    weights_all[key_rm] = weights
                    # weights = [1/(len(input_features) * 2)] * (len(input_features) * 2)
                    # target_feats_ids = [0,1]
                    # rm_worst_output = metric.incremental_output_metric(x_clean, y_clean, test_dataset_generator, best_model, outer_dist=outer_dists, 
                    #                                                    weights=weights, training_type="noise-aware", path=f"{model_path}/rm_results",
                    #                                                    target_feat_ids=target_feats_ids)
                else:
                    if loss_function == "custom_loss":
                        weights = estimate_weights(f"{model_path_i}", input_features, test_dataset_generator, num_samples=x_len, loss_function=loss_function,
                                                   metric=metric, x_noisy=x_noisy, len_input_features=input_shape, bl_ratio=rm_bl_train,
                                                   nominator=gxs_dists, y_clean=y_clean, model_type=config["type"])
                    else:
                        print("model_path_i", model_path_i)
                        weights = estimate_weights(f"{model_path_i}", input_features,test_dataset_generator, num_samples=x_len, model_type=config["type"])
                print("weights", weights)
                print("correct_weights", correct_weights)
                # if all weights are 0, then we use the correct weights
                if np.all(weights == 0):
                    weights = correct_weights
                weights = correct_weights
                
                
                for idx_shape, x_noise_vector in enumerate(x_noisy_new):
                    # x_noise_vector = torch.from_numpy(x_noise_vector).float()
                    y_noise_vector = model.predict(x_noise_vector)
                    
                    # print("y_noise_vector", y_noise_vector)
                    # print("x_noise_vector", x_noise_vector)
                    # exit()
                    # y_noisy_new[idx_shape, :] = y_noise_vector.flatten().cpu().numpy()
                    y_noisy_new[idx_shape, :] = y_noise_vector.flatten()
                rm = metric.calculate_metric(x_clean_orig, y_clean_orig, x_hat=x_noisy_new, y_hat=y_noisy_new, outer_dist=outer_dists, weights=weights, 
                                             path=f"{model_i_res_folder}/xbar_{key_rm}")
                
                # for key_rm in rm.keys():
                #     if rm_worst_output is None:
                #         rm_worst_output = rm
                #     else:
                #        if rm_worst_output[key_rm]['Output distance'] < rm[key_rm]["Output distance"]:
                #            rm_worst_output = rm
                rms[key_rm] = rm
                # TODO: instead of resetting the num_samples, we should create a copy of the dataset generator
                
                test_dataset_generator.num_samples = x_clean_orig.shape[0]
                test_dataset_generator.noise_generator.num_samples = test_dataset_generator.num_samples
           
            ########### save rm to txt filethe thre
            print("results saved in", model_i_res_folder)
            with open(f"{model_i_res_folder}/rm_new.txt", "w") as outfile:
                json.dump(rms, outfile, indent=4)

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
            loss_fuction = "mse"
            res_folder = f"./results_{os.path.splitext(json_file)[0]}/loss_{loss_fuction}/{noise_type}/non-dp/"
            json_path = f'./configs/equations_all/{json_file}'

            main(res_folder, json_path, loss_fuction="mse", noise_type=noise_type)

            # loss_fuction = "custom_loss"
            # res_folder = f"./results_try_{os.path.splitext(json_file)[0]}/loss_{loss_fuction}/{noise_type}/non-dp/"
            # main(res_folder, json_path, loss_fuction="custom_loss", noise_type=noise_type)
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