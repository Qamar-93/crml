from Training.ModelTrainer import ModelTrainer
import tensorflow as tf
input_shape= 3
trainer = ModelTrainer().get_model("cnn", shape_input=input_shape, loss_function="mse")
model = trainer.model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="mse")
# model.load_weights(f"{model_path_i}/model_weights.h5")  
trainer.load_model(model_obj=model, filepath="/home/qamar/workspace/crml/code/results_I_12_2_recent/loss_mse/normal/non-dp/cnn/clean/models_all/model_1")
print("Model loaded successfully")