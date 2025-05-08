import glob
import os
import json
import datetime
import numpy as np
import torch
import tensorflow as tf
import pandas as pd
from math import ceil
from DataGens.NoiseGenerator import NoiseGenerator
from DataGens.DatasetGenerator import DatasetGenerator
from Training.ModelTrainer import ModelTrainer
from utils.training_utils import CustomLoss, CustomMetric
from utils.helpers import evaluate_and_plot, evaluate
from Metric.RobustnessMetric import RobustnessMetric
from Metric.weights_estimation import estimate_weights
from utils.dists import L2_distance

def setup_environment():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    return device

def load_configs(json_path):
    with open(json_path) as f:
        configs = json.load(f)
    return configs

def initialize_data_generators(configs, noise_type, epsilon):
    x_len = configs["num_samples"]
    num_noises = 20
    variance = 0.5
    noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=noise_type, variance=variance, epsilon=epsilon)
    equation_str = configs["equation"]
    input_features = configs["features"]
    dataset_generator = DatasetGenerator(equation_str, noise_model, input_features, num_samples=x_len)
    return dataset_generator, input_features

def prepare_data(config, dataset_generator, metric, input_shape, training_type, num_inputs):
    xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices = dataset_generator.split(config, metric_instance=metric)
    x_noisy, y_noisy = xy_noisy
    x_clean, y_clean = xy_clean
    gx, gx_y = gx_gy
    indices_train, indices_valid = indices

    x_noisy_train = x_noisy[:, indices_train, :]
    x_noisy_valid = x_noisy[:, indices_valid, :]
    y_noisy_train = y_noisy[:, indices_train]
    y_noisy_valid = y_noisy[:, indices_valid]
    x_clean_train = x_clean[indices_train, :]
    x_clean_valid = x_clean[indices_valid, :]
    y_clean_train = y_clean[indices_train]
    y_clean_valid = y_clean[indices_valid]

    if training_type == "noise-aware":
        x_noisy = np.tile(x_noisy, (1, 1, 2))
        x_noisy[:, :, num_inputs:] = 0
        x_noisy[:, :, num_inputs:2*num_inputs] = gx
        input_shape = num_inputs * 2
        xy_noisy = (x_noisy, y_noisy)

    return (xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices), input_shape

def calculate_baseline_metric(dataset_generator, metric, x_clean_train, y_clean_train, x_noisy_train, y_noisy_bl_train, correct_weights, res_folder):
    rm_bl_train = metric.calculate_metric(x_clean_train, y_clean_train, 
                                          x_hat=x_noisy_train, y_hat=y_noisy_bl_train,
                                          outer_dist=["Euclidean"], weights=correct_weights, path=f"{res_folder}/baseline/training")["Euclidean"]
    with open(f"{res_folder}/baseline/training/rm.txt", "w") as outfile:
        outfile.write(str(rm_bl_train))
    return rm_bl_train["Output distance"]

def train_models(config, trainer, xy_train, xy_valid, input_shape, loss_function, models_folder, models_num):
    best_model = None
    best_history = None
    models = []
    losses = []
    valid_losses = []
    valid_losses_all_epochs = []
    last_epoch = []
    rm_vals = []

    for i in range(models_num):
        if not os.path.exists(f"{models_folder}/model_{i+1}"):
            os.makedirs(f"{models_folder}/model_{i+1}")
        model, history = trainer.compile_and_fit(xy_train=xy_train, xy_valid=xy_valid, fit_args=config["fit_args"])
        if best_model is None or history.history['val_loss'][-1] < best_history.history['val_loss'][-1]:
            best_model = model
            best_history = history
        models.append(model)
        losses.append(history.history['loss'][-1])
        valid_losses.append(history.history['val_loss'][-1])
        valid_losses_all_epochs.append(history.history['val_loss'])
        last_epoch.append(len(history.history['loss']))

    return best_model, best_history, models, losses, valid_losses, valid_losses_all_epochs, last_epoch

def save_models(models, models_folder, trainer):
    for i, model in enumerate(models):
        if not os.path.exists(f"{models_folder}/model_{i+1}"):
            os.makedirs(f"{models_folder}/model_{i+1}")
        trainer.save_model(model, f"{models_folder}/model_{i+1}")

def save_losses(models_folder, losses, valid_losses, last_epoch, rm_vals, valid_losses_all_epochs):
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

def robustness_testing(models, configs, input_features, res_folder, noise_type, epsilon):
    x_len = configs["num_samples"]
    num_noises = 40
    variance = 0.3
    equation_str = configs["equation"]
    test_noise_model = NoiseGenerator(num_samples=x_len, num_noises=num_noises, noise_type=noise_type, variance=variance, epsilon=epsilon)
    test_dataset_generator = DatasetGenerator(equation_str, test_noise_model, input_features, num_samples=x_len)
    x_clean_orig, y_clean_orig = test_dataset_generator.generate_dataset()
    y_clean_orig = y_clean_orig.ravel()
    test_dataset_generator.num_samples = x_clean_orig.shape[0]
    test_dataset_generator.noise_generator.num_samples = test_dataset_generator.num_samples
    no_noisy_tests = 1
    random_seeds_all = [np.linspace(0, 1000, len(input_features), dtype=int) for _ in range(no_noisy_tests)]
    num_inputs = len(input_features)

    for idx, model in enumerate(models):
        config = configs["models"][0]
        model_i_res_folder = f"{res_folder}/models_all/rm_results_{noise_type}/model_{idx+1}"
        if not os.path.exists(model_i_res_folder):
            os.makedirs(model_i_res_folder)
        x_noisy_all = {j: None for j in range(no_noisy_tests)}
        y_noisy_all = {k: None for k in range(no_noisy_tests)}
        weights_all = {l: None for l in range(no_noisy_tests)}
        rms = {rm_i: None for rm_i in range(no_noisy_tests)}
        target_feats_ids = [el for el in range(len(input_features))]

        for i_noisy in range(no_noisy_tests):
            x_noisy, y_noisy = test_dataset_generator.modulate_clean(x_clean_orig, y_clean_orig, target_feat_idx=target_feats_ids, random_seeds=random_seeds_all[i_noisy])
            x_noisy_all[i_noisy] = x_noisy
            y_noisy_all[i_noisy] = y_noisy

        for key_rm, value in x_noisy_all.items():
            x_noisy = value
            y_noisy = y_noisy_all[key_rm]
            y_noisy_new = np.zeros((y_noisy.shape[0], y_noisy.shape[1]))

            if config["training_type"] == "noise-aware":
                x_noisy_new = np.zeros((x_noisy.shape[0], x_noisy.shape[1], x_noisy.shape[2] * 2))
                for idx_shape in range(x_noisy.shape[2]):
                    x_noisy_new[:, :, idx_shape] = x_noisy[:, :, idx_shape]
                    x_noisy_new[:, :, idx_shape + x_noisy.shape[2]] = gx[:, idx_shape]
            else:
                x_noisy_new = x_noisy

            for idx_shape, x_noise_vector in enumerate(x_noisy_new):
                y_noise_vector = model.predict(x_noise_vector)
                y_noisy_new[idx_shape, :] = y_noise_vector.flatten()
            metric = RobustnessMetric()
            model_path = f"{config['model_path']}/model_{idx+1}"
            weights = estimate_weights(model_path=model_path, inputs=input_features, dataset_generator=test_dataset_generator, num_samples=1000, model_type=config["type"])
            rm = metric.calculate_metric(x_clean_orig, y_clean_orig, x_hat=x_noisy_new, y_hat=y_noisy_new, outer_dist=["Euclidean"], weights=weights, path=f"{model_i_res_folder}/xbar_{key_rm}")
            rms[key_rm] = rm

        with open(f"{model_i_res_folder}/rm_new.txt", "w") as outfile:
            json.dump(rms, outfile, indent=4)

def main(res_folder, json_path, loss_function, noise_type, epsilon=0.5):
    device = setup_environment()
    configs = load_configs(json_path)
    dataset_generator, input_features = initialize_data_generators(configs, noise_type, epsilon)
    metric = RobustnessMetric()
    input_shape = len(input_features)

    for config in configs["models"]:
        training_type = config["training_type"]
        models_folder = f"{res_folder}/{config['type']}/{config['training_type']}/models_all"
        plots_folder = f"{res_folder}/{config['type']}/{config['training_type']}/plots"
        extended_data = False

        (xy_train, xy_valid, xy_test, xy_noisy, xy_clean, gx_gy, indices), input_shape = prepare_data(config, dataset_generator, metric, input_shape, training_type, len(input_features))

        y_noisy_bl_train = np.zeros((xy_noisy[1].shape[0], xy_noisy[1].shape[1]))
        for idx_shape, x_noise_vector in enumerate(xy_noisy[0]):
            y_noise_vector = dataset_generator.apply_equation(x_noise_vector[:, :len(input_features)])
            y_noisy_bl_train[idx_shape, :] = y_noise_vector.flatten()

        correct_weights = estimate_weights(model_path=f"{res_folder}/baseline/", inputs=input_features, dataset_generator=dataset_generator,
                                           num_samples=1000, model_type="expression")
        if training_type == "noise-aware":
            correct_weights = np.append(correct_weights, np.zeros(len(correct_weights)))
        rm_bl_train = calculate_baseline_metric(dataset_generator, metric, xy_clean[0], xy_clean[1], xy_noisy[0], y_noisy_bl_train, correct_weights, res_folder)

        trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
        print(config["load"])
        if config["load"] == True:
            print("Loading models")
            models = []
            losses_files = glob.glob(f"{models_folder}/losses_*.txt")
            losses_files.sort()
            if losses_files:
                first_loss_file = losses_files[0]
                losses = np.loadtxt(first_loss_file)
            for i in range(10):
                model_path_i = f"{models_folder}/model_{i+1}"
                trainer = ModelTrainer().get_model(config["type"], shape_input=input_shape, loss_function=loss_function)
                model = trainer.model
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                model.compile(optimizer=optimizer, loss=loss_function)
                trainer.load_model(model_obj=model, filepath=f"{model_path_i}")
                models.append(model)
            best_model = models[np.argmin(losses)]
        else:
            print("Training models")
            best_model, best_history, models, losses, valid_losses, valid_losses_all_epochs, last_epoch = train_models(config, trainer, xy_train, xy_valid, input_shape, loss_function, models_folder, 10)
            save_models(models, models_folder, trainer)
            save_losses(models_folder, losses, valid_losses, last_epoch, [], valid_losses_all_epochs)

        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        robustness_testing(models, configs, input_features, res_folder, noise_type, epsilon)

if __name__ == '__main__':
    json_files = [f for f in os.listdir('./configs/equations_all/') if f.endswith('.json')]
    json_files = ["IV_9.json"]

    for json_file in json_files:
        res_folder = f"./results_mse_dp_{os.path.splitext(json_file)[0]}"
        noise_type = "normal"
        if noise_type == "normal":
            loss_function = "mse"
            res_folder = f"./results_{os.path.splitext(json_file)[0]}_recent/loss_{loss_function}/{noise_type}/non-dp/"
            json_path = f'./configs/equations_all/{json_file}'
            main(res_folder, json_path, loss_function="mse", noise_type=noise_type)
