import matplotlib.pyplot as plt

# Read the data from the text file as columns
import pandas as pd
# Plot the data
def plot(df_custom, df_mse, equation, noise_type, dp_type, model_number, training_data, epsilon=0):
    list_of_columns = [0, 1, 2]    
    plt.figure()
    plt.clf()
    plt.plot(df_custom[list_of_columns[0]], df_custom[list_of_columns[1]], label = 'True Solution', color = 'blue')
    plt.plot(df_custom[list_of_columns[0]], df_custom[list_of_columns[2]], label = 'G(Y) - custom', color = 'red')
    plt.plot(df_mse[list_of_columns[0]], df_mse[list_of_columns[2]], label = 'G(Y) - mse', color = 'green')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title('Data Plot')
    plt.legend()
    if epsilon != 0:
        plt.savefig(f'{equation}_{noise_type}_{dp_type}_{model_number}_{training_data}_epsilon_{epsilon}.png')
        # save txt with columns: x, y, y_custom, y_mse
        df = pd.DataFrame({'x': df_custom[list_of_columns[0]], 'y': df_custom[list_of_columns[1]], 'y_custom': df_custom[list_of_columns[2]], 'y_mse': df_mse[list_of_columns[2]]})
        df.to_csv(f'./plots_checking/{equation}_{noise_type}_{dp_type}_{model_number}_{training_data}_epsilon_{epsilon}.txt', index=False)
    else:
        plt.savefig(f'./plots_checking/{equation}_{noise_type}_{dp_type}_{model_number}_{training_data}.png')
        df = pd.DataFrame({'x': df_custom[list_of_columns[0]], 'y': df_custom[list_of_columns[1]], 'y_custom': df_custom[list_of_columns[2]], 'y_mse': df_mse[list_of_columns[2]]})
        df.to_csv(f'./plots_checking/{equation}_{noise_type}_{dp_type}_{model_number}_{training_data}.txt', index=False)
        
        
if __name__ == '__main__':
    equation = 'I_6_2a'
    results_folder = 'results_' + equation
    noise_type = 'laplace'
    dp_type = 'non-dp'
    model_number = 'model_2'
    # training_data = 'noise-aware'
    training_data = 'clean'
    # Read the file
    # /home/qamar/workspace/crml/code/results_I_6_2a/loss_mse/laplace_dp/epsilon_0.1/linear/clean/models_all
    df_custom = None
    df_mse = None
    for model_number in ['model_1', 'model_2']:
        for dp_type in ['dp', 'non-dp']:
            for noise_type in ['laplace', 'normal']:
                if noise_type == 'laplace' and dp_type == 'dp':
                    for epsilon in [0.1, 1, 10]:
                        for training_data in ['clean', 'noise-aware']:
                            df_custom = pd.read_csv(f'./{results_folder}/loss_custom_loss/{noise_type}_dp/epsilon_{epsilon}/linear/{training_data}/models_all/rm_results_laplace/{model_number}/xbar_0/G_output/G for output.txt', delimiter = ",", header = None)
                            df_mse = pd.read_csv(f'./{results_folder}/loss_mse/{noise_type}_dp/epsilon_{epsilon}/linear/{training_data}/models_all/rm_results_laplace/{model_number}/xbar_0/G_output/G for output.txt', delimiter = ",", header = None)

                            plot(df_custom, df_mse, equation, noise_type, dp_type, model_number, training_data, epsilon)
                        
                elif noise_type == 'laplace' and dp_type == 'non-dp':
                    for epsilon in [0.1, 1, 10]:
                        for training_data in ['clean', 'noise-aware']:
                            df_custom = pd.read_csv(f'./{results_folder}/loss_custom_loss/{noise_type}/epsilon_{epsilon}/linear/{training_data}/models_all/rm_results_laplace/{model_number}/xbar_0/G_output/G for output.txt', delimiter = ",", header = None)
                            df_mse = pd.read_csv(f'./{results_folder}/loss_mse/{noise_type}/epsilon_{epsilon}/linear/{training_data}/models_all/rm_results_laplace/{model_number}/xbar_0/G_output/G for output.txt', delimiter = ",", header = None)
                            plot(df_custom, df_mse, equation, noise_type, dp_type, model_number, training_data, epsilon)
                else:
                    for training_data in ['clean', 'noise-aware']:
                        df_custom = pd.read_csv(f'./{results_folder}/loss_custom_loss/{noise_type}/{dp_type}/linear/{training_data}/models_all/rm_results_laplace/{model_number}/xbar_0/G_output/G for output.txt', delimiter = ",", header = None)

                        df_mse = pd.read_csv(f'./{results_folder}/loss_mse/{noise_type}/{dp_type}/linear/{training_data}/models_all/rm_results_laplace/{model_number}/xbar_0/G_output/G for output.txt', delimiter = ",", header = None)
                        plot(df_custom, df_mse, equation, noise_type, dp_type, model_number, training_data, epsilon=0)