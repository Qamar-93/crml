import pickle
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from Training.ModelTrainer import ModelTrainer
from Training.symbolic_network import SymbolicNetL0
from utils import functions
import keras
from utils.training_utils import CustomLoss

def estimate_weights(model_path, inputs, dataset_generator, training_type="clean", num_samples=100, model_type="Linear", loss_function="mse", **kwargs):
    """
    Estimate the weights of the input features.

    Parameters:
    model (object): path to the trained model.
    inputs (obj): The input data, which is a dictionary of the form {feature_name: {range: [min, max], type: data_type}}.
    Returns:
    ndarray: The estimated weights of the input features.
    """
    
    if training_type == "noise-aware":
        # for noise-aware training, we need to add the noise-aware noise as a feature for each input in inputs with value 0, also the distance between g and x should be 0 as a new feature
        new_inputs = inputs.copy()
        for key in inputs.keys():
            new_key = "g" + key
            new_inputs[new_key] = inputs[key]            
    else:
            new_inputs = inputs.copy()
        
    input_feats = list(new_inputs.items())
     
    problem = {
        'num_vars': len(new_inputs),
        'names': [f"x{i}" for i in range(len(new_inputs))],
        'bounds': [[val["range"][0], val["range"][1]] for key, val in input_feats]
    }
    
    param_values = saltelli.sample(problem, 10000, calc_second_order=False)
    
    if training_type == "noise-aware":
        param_values[:, -len(inputs):] = 0

    if model_type == "CustomModel":
        from Training.CustomModel import CustomModel
        model = CustomModel(input_shape=len(inputs), loss_function="mean_squared_error", output_shape=1)
        # model = CustomModel.load_model(model_path, input_shape=(len(new_inputs), 2), loss_function="mean_squared_error", output_shape=1)
        model = keras.models.load_model(model_path)

    
    elif model_type == "expression":
        model = dataset_generator
        
    elif model_type == "SymbolicNetL0":
        with open(f"{model_path}/{model_path.split('/')[-1]}.pickle", "rb") as f:
            loaded_model_data = pickle.load(f)
        loaded_weights = loaded_model_data["weights"]
        
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        initial_weights = [torch.tensor(w).to(device) for w in loaded_weights]
        
        model = SymbolicNetL0(2, in_dim=len(input_feats), funcs=functions.default_func, initial_weights=initial_weights).to(device)

    else:
        if loss_function == "custom_loss":
            # read the custom loss parameters from kwargs
            metric = kwargs["metric"]
            x_noisy = kwargs["x_noisy"]
            bl_ratio = kwargs["bl_ratio"]
            y_clean = kwargs["y_clean"]

            trainer = ModelTrainer().get_model(model_type, shape_input=len(input_feats)
                                               , loss_function=loss_function)
            model = trainer.model
            customloss = CustomLoss(model=model, metric=metric, y_clean=y_clean, x_noisy=x_noisy, len_input_features=len(input_feats), bl_ratio=bl_ratio)

            optimizer = keras.optimizers.Adam(learning_rate=0.0001)

            model.compile(optimizer='adam', loss=customloss)
            # model.load_weights(f"{model_path}/model_weights.h5")
            trainer.load_model(model_obj=model, filepath=f"{model_path}")

            # model = keras.models.load_model(model_path, compile=False)
            
            # customloss = CustomLoss(model=model, metric=metric, y_clean=y_clean, x_noisy=x_noisy, len_input_features=len_input_features, bl_ratio=bl_ratio, nominator=nominator)
            # model.compile(optimizer='adam', loss=customloss)
        else:
            print("input feats", input_feats)
            trainer = ModelTrainer().get_model(model_type, shape_input=len(input_feats)
                                               ,loss_function=loss_function)
            model = trainer.model
            print("model_type", model_type)
            if model_type=="linear":
                optimizer = keras.optimizers.Adam(learning_rate=0.0001)
                model.compile(optimizer=optimizer, loss=loss_function)
            # model.load_weights(f"{model_path}/model_weights.h5")
            if model_type=="linear":
                trainer.load_model(model_obj=model, filepath=f"{model_path}")
                model = trainer.model
            elif model_type=="cnn":
                optimizer = keras.optimizers.Adam(learning_rate=0.0001)
                model.compile(optimizer=optimizer, loss=loss_function)
                trainer.load_model(model_obj=model, filepath=f"{model_path}")
                model = trainer.model
            else:
                model = trainer.load_model(model_obj=model, filepath=f"{model_path}")

            # model = keras.models.load_model(model_path)
    # param_values = np.reshape(param_values, (param_values.shape[0], len(new_inputs)))

    if model_type == "expression":
        Y = model.apply_equation(param_values).flatten()
    elif model_type == "SymbolicNetL0":
        print("param_values", param_values.shape)
        Y = model.predict(torch.tensor(param_values, dtype=torch.float)).to(device)
        Y = Y.cpu().detach().numpy().flatten()
    else:
        Y = model.predict(param_values).flatten()

    # Run model
    Si = sobol.analyze(problem, Y, calc_second_order=False)

    weights = Si["ST"]
    # if the sum of the weights is not 1, then normalize the weights
    if np.sum(weights) == 0:
        return weights
        
    elif np.sum(weights) != 1:
        weights = weights / np.sum(weights)
    
        return weights


