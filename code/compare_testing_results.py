


import matplotlib.pyplot as plt
import pandas as pd    
import os
def read_rm(equation, loss, dp_type, training_distribution, model_type, models_set, distance, epsilon=None):
    rms =[]
    path = ""
    for u_model_idx, u_model in enumerate(models_set):
        if training_distribution == "normal":
            path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_{loss}/{training_distribution}/{dp_type}/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_1/rm_new.txt"
        elif training_distribution == "laplace" or training_distribution == "laplace_dp":
            if model_type == "linear":
                path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_{loss}/{training_distribution}/epsilon_{epsilon}/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_1/rm_new.txt"
            else:
                path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_mse/normal/non-dp/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_1/rm_new.txt"

        with open(path, "r") as f:
        #read json object
            rm = eval(f.read())
            print(rm, "for", path)
            rms.append(rm["0"][distance]["Ratio"])
    print("rms_not_cumulative: ", rms)
    non_cumulative_rms = rms
    # accumulative list of rms such that [1,2,3] becomes [1,3,6]
    rms = [sum(rms[:i+1]) for i in range(len(rms))]
    print("rms_cumulative: ", rms)

    return rms, non_cumulative_rms

def read_rm_tsr(equation, loss, dp_type, training_distribution, model_type, models_set, distance, epsilon=None):
    rms =[]
    path = ""
    for u_model_idx, u_model in enumerate(models_set):
        # if training_distribution == "normal":
            # path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_{loss}/{training_distribution}/{dp_type}/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_1/rm_new.txt"
        path = f"/data/qamar/symbolicregression_results/results_{equation}_recent/loss_{loss}/{training_distribution}/non-dp/est_results/model_1/{u_model}/robustness_metric.txt"
        # elif training_distribution == "laplace" or training_distribution == "laplace_dp":
        #     if model_type == "linear":
        #         path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_{loss}/{training_distribution}/epsilon_{epsilon}/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_1/rm_new.txt"
        #     else:
        #         path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_mse/normal/non-dp/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_1/rm_new.txt"

        with open(path, "r") as f:
        #read json object
            rm = eval(f.read())
            print(rm, "for", path)
            rms.append(rm["L2"]["Ratio"])
    print("rms_not_cumulative: ", rms)
    non_cumulative_rms = rms
    # accumulative list of rms such that [1,2,3] becomes [1,3,6]
    rms = [sum(rms[:i+1]) for i in range(len(rms))]
    print("rms_cumulative: ", rms)

    return rms, non_cumulative_rms

def read_mse(equation, loss, dp_type, training_distribution, model_type, models_set, distance, epsilon=None):
    mse =[]
    path = ""
    for u_model_idx, u_model in enumerate(models_set):
        if training_distribution == "normal":
            path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_{loss}/{training_distribution}/{dp_type}/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_2/mse_noisy_new.txt"
        elif training_distribution == "laplace" or training_distribution == "laplace_dp":
            if model_type == "linear":
                path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_{loss}/{training_distribution}/epsilon_{epsilon}/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_2/mse_noisy_new.txt"
            else:
                path = f"/home/qamar/workspace/crml/code/results_{equation}/loss_mse/normal/non-dp/{model_type}/clean/models_all/rm_results_nmodel_{u_model}_new/model_2/mse_noisy_new.txt"

        with open(path, "r") as f:
        #read json object
            error = eval(f.read())
            mse.append(error["0"])
    # accumulative list of mse such that [1,2,3] becomes [1,3,6]
    # mse = [sum(mse[:i+1]) for i in range(len(mse))]
    print("mse_cumulative: ", mse)
    return mse
models_names_text = {
    "linear": "Linear",
    "RF": "RF",
    "LR": "LR",
    }
if __name__ == "__main__":    
    
    # paths = [
    #     "/home/qamar/workspace/crml/code/predicted_vs_true_linear_mse_normal.txt",
    #     "/home/qamar/workspace/crml/code/predicted_vs_true_LR_mse_normal.txt",
    #     "/home/qamar/workspace/crml/code/predicted_vs_true_RF_mse_normal.txt",
    #     "/home/qamar/workspace/crml/code/predicted_vs_true_linear_custom_loss_normal.txt"
    # ]
    # df_all = pd.DataFrame()
    # for idx, path in enumerate(paths):
    #     df = pd.read_csv(path, delimiter=",")
    #     if idx == 0:
    #         df_all["y_true"] = df["y_true"]
    #         df_all["x"] = df["x"]
    #     model_loss = path.split("true_")[1].split("_normal")[0]    
    #     df_all[f"y_pred_{model_loss}"] = df["y_pred"]
    # # save to new csv
    # df_all.to_csv("/home/qamar/workspace/crml/code/predicted_vs_true_all_models.csv", header=True, index=False)
    # print(df_all)
    # plt.plot(df_all["y_true"], df_all["y_true"], label="True")
    # plt.plot(df_all["y_true"], df_all["y_pred_linear_mse"], label="Linear - MSE")
    # plt.plot(df_all["y_true"], df_all["y_pred_LR_mse"], label="LR - MSE")
    # plt.plot(df_all["y_true"], df_all["y_pred_RF_mse"], label="RF - MSE")
    # plt.plot(df_all["y_true"], df_all["y_pred_linear_custom_loss"], label="Linear - MSE + Penalty")
    # plt.legend()
    # plt.savefig("./predicted_vs_true_all_models.png")         
    
    equation = "I_6_2a"
    loss = "mse"
    dp_type = "non-dp"
    training_distribution = "laplace"
    noise_model = "n0"
    model_type = "linear"
    # path = "/home/qamar/workspace/crml/code/results_I_6_2a/loss_mse/normal/dp/baseline/nmodel_n0/rm.txt"
    # noise models: n0, n1, n2, n3, n4 are normal with different variances 0.1, 0.4, 0.6, 0.8, 1
    # noise models: n5, n6, n7, n8, n9 are uniform with different variances
    # noise models: n10, n11, n12, n13, n14, n15, n16, n17 are laplace with different epsilons 0.1, 1, 2, 4, 6, 8, 10, 20

    variances = [0.1, 0.4, 0.6, 0.8, 1]
    epsilons = [0.1, 1, 2, 4, 6, 8, 10, 20]

    normal_models = ["n0", "n1", "n2", "n3", "n4"]
    uniform_models = ["n5", "n6", "n7", "n8", "n9"]
    laplace_models = ["n10", "n11", "n12", "n13", "n14", "n15", "n16", "n17"]
    pareto_models = ["n18", "n19", "n20", "n21", "n22"]
    laplace_non_dp_models = ["n23", "n24", "n25", "n26", "n27"]
    normal_models_percentages = ["n28", "n29", "n30", "n31", "n32", "n33", "n34", "n35", "n36", "n37"]
    uniform_models_percentages = ["n38", "n39", "n40", "n41", "n42", "n43", "n44", "n45", "n46", "n47"]
    laplace_models_percentages = ["n48", "n49", "n50", "n51", "n52", "n53", "n54", "n55", "n56", "n57"]
    laplace_models_variances = ["n23", "n24", "n25", "n26", "n27"]
    training_epsilons = [0.1, 1, 10]
    
    
    # compare the linear model - mse, with different distance functions
    
    
    training_distribution = "normal"
    
    def compare_distances(equation, loss, dp_type, training_distribution, model_type, normal_models, figure_title):
        eu_linear_mse = read_rm(equation, loss, dp_type, training_distribution, model_type, normal_models, "Euclidean")
        l1_linear_mse = read_rm(equation, loss, dp_type, training_distribution, model_type, normal_models, "L1")
        f_linear_mse = read_rm(equation, loss, dp_type, training_distribution, model_type, normal_models, "Frechet")
        h_linear_mse = read_rm(equation, loss, dp_type, training_distribution, model_type, normal_models, "Hausdorff")
        
        plt.clf()
        plt.plot(variances, eu_linear_mse, label="Euclidean")
        plt.plot(variances, l1_linear_mse, label="L1")
        plt.plot(variances, f_linear_mse, label="Frechet")
        plt.plot(variances, h_linear_mse, label="Hausdorff")
        plt.legend()
        plt.xlabel("Noise Variance")
        plt.ylabel("RM")
        
        plt.title(figure_title)
        path = "./distance_functions/"
        if os.path.exists(path) == False:
            os.makedirs(path)
            
        plt.savefig(f"./{path}//rm_vs_noise_variance_{model_type}_{training_distribution}_distnaces_{loss}.png")
    
    # compare_distances(equation, loss, dp_type, training_distribution, model_type, normal_models, "RM vs Noise Variance - Linear Model - MSE")
    # compare_distances(equation, "custom_loss", dp_type, training_distribution, model_type, normal_models, "RM vs Noise Variance - Linear Model - MSE+penalty")
    # compare_distances(equation, loss, dp_type, training_distribution, "RF", normal_models, "RM vs Noise Variance - RF")
    # compare_distances(equation, loss, dp_type, training_distribution, "LR", normal_models, "RM vs Noise Variance - LR")

    def compare_testing_distributions(equation, loss, dp_type, training_distribution, model_type, normal_models, uniform_models, laplace_models, pareto_models, distance):
        normal_rm = read_rm(equation, loss, dp_type, training_distribution, model_type, normal_models, distance)
        uniform_rm = read_rm(equation, loss, dp_type, training_distribution, model_type, uniform_models, distance)
        laplace_rm = read_rm(equation, loss, dp_type, training_distribution, model_type, laplace_models, distance)
        preto_rm = read_rm(equation, loss, dp_type, training_distribution, model_type, pareto_models, distance)
        
        
        plt.clf()
        plt.plot(variances, normal_rm, label="Normal Distribution")
        
        plt.plot(variances, uniform_rm, label="Uniform Distribution")
        plt.plot(variances, laplace_rm, label="Laplace Distribution")
        # plt.plot(variances, preto_rm, label="Pareto Distribution")
        plt.legend()
        plt.xlabel("Noise Variance")
        plt.ylabel("RM")
        
        plt.title(f"RM vs Noise Variance - {models_names_text[model_type]} Model - {training_distribution} - {distance}")
        path = "./testing_distributions/"
        if os.path.exists(path) == False:
            os.makedirs(path)
        plt.savefig(f"./{path}//rm_vs_noise_variance_{model_type}_{training_distribution}_{distance}_distributions_{loss}.png")
    
    training_distribution = "normal"
    # compare_testing_distributions(equation, loss, dp_type, training_distribution, 'linear', normal_models, uniform_models, laplace_non_dp_models, pareto_models, "Euclidean")
    # compare_testing_distributions(equation, "custom_loss", dp_type, training_distribution, 'linear', normal_models, uniform_models, laplace_non_dp_models, pareto_models, "Euclidean")
    # compare_testing_distributions(equation, loss, dp_type, training_distribution, 'RF', normal_models, uniform_models, laplace_non_dp_models,  pareto_models, "Euclidean")
    # compare_testing_distributions(equation, loss, dp_type, training_distribution, 'LR', normal_models, uniform_models, laplace_non_dp_models,  pareto_models, "Euclidean")
    
    # compare models with same distribution and different models
    def compare_models(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance, testing_distribution):
        
        rm_linear,_ = read_rm(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance)
        rm_linear_custom,_ = read_rm(equation, "custom_loss", dp_type, training_distribution, model_type, distribution_models, distance)
        rm_RF,_ = read_rm(equation, loss, dp_type, training_distribution, "RF", distribution_models, distance)
        rm_LR,_ = read_rm(equation, loss, dp_type, training_distribution, "LR", distribution_models, distance)
        rm_cnn,_ = read_rm(equation, loss, dp_type, training_distribution, "cnn", distribution_models, distance)
        rm_cnn_custom,_ = read_rm(equation, "custom_loss", dp_type, training_distribution, "cnn", distribution_models, distance)
        rm_tsr, _ = read_rm_tsr(equation, loss, dp_type, training_distribution, "tsr", distribution_models, distance)
        
        mse_linear = read_mse(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance)
        mse_linear_custom = read_mse(equation, "custom_loss", dp_type, training_distribution, model_type, distribution_models, distance)
        mse_RF = read_mse(equation, loss, dp_type, training_distribution, "RF", distribution_models, distance)
        mse_LR = read_mse(equation, loss, dp_type, training_distribution, "LR", distribution_models, distance)
        mse_cnn = read_mse(equation, loss, dp_type, training_distribution, "cnn", distribution_models, distance)
        mse_cnn_custom = read_mse(equation, "custom_loss", dp_type, training_distribution, "cnn", distribution_models, distance)
        plt.clf()
        plt.plot(variances, rm_linear, label="Linear Model")
        plt.plot(variances, rm_linear_custom, label="Linear Model - MSE + Penalty")
        plt.plot(variances, rm_RF, label="RF")
        plt.plot(variances, rm_LR, label="LR")
        plt.plot(variances, rm_cnn, label="CNN")
        plt.plot(variances, rm_tsr, label="TSR")
        plt.plot(variances, rm_cnn_custom, label="CNN - MSE + Penalty")
        plt.legend()
        plt.xlabel("Noise Variance")
        plt.ylabel("RM")
        plt.title(f"RM vs Noise Variance - {models_names_text[model_type]} Model - {training_distribution} - {distance} - {testing_distribution}")
        path = "./tsr_testing_distributions/"
        if os.path.exists(path) == False:
            os.makedirs(path)
        plt.savefig(f"./{path}/rm_vs_noise_variance_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
        print("figures is saved in ", f"./{path}/rm_vs_noise_variance_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
        df = pd.DataFrame()
        df["variances"] = variances
        df["rm-linear"] = rm_linear
        df["rm-linear_custom"] = rm_linear_custom
        df["rm-RF"] = rm_RF
        df["rm-LR"] = rm_LR
        df["rm-cnn"] = rm_cnn
        df["rm-cnn-custom"] = rm_cnn_custom
        df["rm-tsr"] = rm_tsr
        df["mse-linear"] = mse_linear
        df["mse-linear_custom"] = mse_linear_custom
        df["mse-RF"] = mse_RF
        df["mse-LR"] = mse_LR
        df["mse-cnn"] = mse_cnn
        df["mse-cnn-custom"] = mse_cnn_custom
        df.to_csv(f"./{path}/rm-vs-noise-variance-{model_type}-{training_distribution}-{distance}-models-{loss}-testing-dist-{testing_distribution}.txt", header=True, index=False)
        
    # training_distribution = "normal"
    
        plt.clf()
        plt.plot(variances, mse_linear, label="Linear Model")
        plt.plot(variances, mse_linear_custom, label="Linear Model - MSE + Penalty")
        plt.plot(variances, mse_RF, label="RF")
        plt.plot(variances, mse_LR, label="LR")
        plt.plot(variances, mse_cnn, label="CNN")
        plt.plot(variances, mse_cnn_custom, label="CNN - MSE + Penalty")
        plt.legend()
        plt.ylabel("MSE")
        plt.xlabel("Noise Variance")
        plt.title(f"MSE vs Noise Variance - {models_names_text[model_type]} Model - {training_distribution} - {distance} - {testing_distribution}")
        plt.savefig(f"./{path}/mse_vs_noise_variance_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
    
    compare_models(equation, loss, dp_type, training_distribution, model_type, normal_models, "Euclidean", "normal")
    
    compare_models(equation, loss, dp_type, training_distribution, model_type, uniform_models, "Euclidean", "uniform")
    
    compare_models(equation, loss, dp_type, training_distribution, model_type, laplace_non_dp_models, "Euclidean", "laplace")
    def compare_models_percentages(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance, testing_distribution):
            
            rm_linear, rm_linear_non_cum = read_rm(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance)
            rm_linear_custom, rm_linear_custom_non_cum = read_rm(equation, "custom_loss", dp_type, training_distribution, model_type, distribution_models, distance)
            rm_RF, rm_RF_non_cum = read_rm(equation, loss, dp_type, training_distribution, "RF", distribution_models, distance)
            rm_LR,rm_LR_non_cum = read_rm(equation, loss, dp_type, training_distribution, "LR", distribution_models, distance)
            rm_cnn,rm_cnn_non_cum = read_rm(equation, loss, dp_type, training_distribution, "cnn", distribution_models, distance)
            rm_tsr, rm_tsr_non_cum = read_rm_tsr(equation, loss, dp_type, training_distribution, "tsr", distribution_models, distance)
            rm_cnn_custom, rm_cnn_custom_non_cum = read_rm(equation, "custom_loss", dp_type, training_distribution, "cnn", distribution_models, distance)
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
            plt.clf()
            plt.plot(percentages, rm_linear, label="Linear Model")
            plt.plot(percentages, rm_linear_custom, label="Linear Model - MSE + Penalty")
            plt.plot(percentages, rm_RF, label="RF")
            plt.plot(percentages, rm_LR, label="LR")
            plt.plot(percentages, rm_cnn, label="CNN")
            plt.plot(percentages, rm_tsr, label="TSR")
            plt.plot(percentages, rm_cnn_custom, label="CNN - MSE + Penalty")
            plt.legend()
            plt.xlabel("Percentage")
            plt.ylabel("RM")
            plt.title(f"RM vs Percentage - {models_names_text[model_type]} Model - {training_distribution} - {distance} - {testing_distribution}")
            # path = "./testing_distributions/"
            path = "./tsr_testing_distributions/"
        
            if os.path.exists(path) == False:
                os.makedirs(path)
            plt.savefig(f"./{path}/rm_vs_percentage_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
            print("figures is saved in ", f"./{path}/rm_vs_percentage_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
            
            plt.clf()
            plt.plot(percentages, rm_linear_non_cum, label="Linear Model")
            plt.plot(percentages, rm_linear_custom_non_cum, label="Linear Model - MSE + Penalty")
            plt.plot(percentages, rm_RF_non_cum, label="RF")
            plt.plot(percentages, rm_LR_non_cum, label="LR")
            plt.plot(percentages, rm_cnn_non_cum, label="CNN")
            plt.plot(percentages, rm_tsr_non_cum, label="TSR")
            plt.plot(percentages, rm_cnn_custom_non_cum, label="CNN - MSE + Penalty")
            plt.legend()
            plt.xlabel("Percentage")
            plt.ylabel("RM")
            plt.title(f"RM vs Percentage - {models_names_text[model_type]} Model - {training_distribution} - {distance} - {testing_distribution}")
            
            plt.savefig(f"./{path}/rm_vs_percentage_non_cumulative_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
            
            df = pd.DataFrame()
            df["percentages"] = percentages
            df["rm-linear"] = rm_linear
            df["rm-linear_custom"] = rm_linear_custom
            df["rm-RF"] = rm_RF
            df["rm-LR"] = rm_LR
            df["rm-cnn"] = rm_cnn
            df["rm-tsr"] = rm_tsr
            df["rm-cnn-custom"] = rm_cnn_custom
            df.to_csv(f"./{path}/rm-vs-percentage-{model_type}-{training_distribution}-{distance}-models-{loss}-testing-dist-{testing_distribution}.txt", header=True, index=False)
            
            mse_linear = read_mse(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance)
            mse_linear_custom = read_mse(equation, "custom_loss", dp_type, training_distribution, model_type, distribution_models, distance)
            mse_RF = read_mse(equation, loss, dp_type, training_distribution, "RF", distribution_models, distance)
            mse_LR = read_mse(equation, loss, dp_type, training_distribution, "LR", distribution_models, distance)
            mse_cnn = read_mse(equation, loss, dp_type, training_distribution, "cnn", distribution_models, distance)
            mse_cnn_custom = read_mse(equation, "custom_loss", dp_type, training_distribution, "cnn", distribution_models, distance)
            # mse_tsr = read_mse(equation, loss, dp_type, training_distribution, "tsr", distribution_models, distance)
            plt.clf()
            plt.plot(percentages, mse_linear, label="Linear Model")
            plt.plot(percentages, mse_linear_custom, label="Linear Model - MSE + Penalty")
            plt.plot(percentages, mse_RF, label="RF")
            plt.plot(percentages, mse_LR, label="LR")
            plt.plot(percentages, mse_cnn, label="CNN")
            plt.plot(percentages, mse_cnn_custom, label="CNN - MSE + Penalty")
            plt.legend()
            plt.ylabel("MSE")
            plt.xlabel("Percentage")
            plt.title(f"MSE vs Percentage - {models_names_text[model_type]} Model - {training_distribution} - {distance} - {testing_distribution}")
            plt.savefig(f"./{path}/mse_vs_percentage_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
            mse_df = pd.DataFrame()
            mse_df["percentages"] = percentages
            mse_df["mse-linear"] = mse_linear
            mse_df["mse-linear_custom"] = mse_linear_custom
            mse_df["mse-RF"] = mse_RF
            mse_df["mse-LR"] = mse_LR
            mse_df["mse-cnn"] = mse_cnn
            mse_df["mse-cnn-custom"] = mse_cnn_custom
            # mse_df["mse-tsr"] = mse_tsr
            mse_df.to_csv(f"./{path}/mse-vs-percentage-{model_type}-{training_distribution}-{distance}-models-{loss}-testing-dist-{testing_distribution}.txt", header=True, index=False)
            
    compare_models_percentages(equation, loss, dp_type, training_distribution, model_type, normal_models_percentages, "Euclidean", "normal")
    compare_models_percentages(equation, loss, dp_type, training_distribution, model_type, uniform_models_percentages, "Euclidean", "uniform")
    compare_models_percentages(equation, loss, dp_type, training_distribution, model_type, laplace_models_percentages, "Euclidean", "laplace")
    exit()    
    def compare_models_mse_vs_rm(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance, testing_distribution):
            
            rm_linear, rm_linear_non_cum = read_rm(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance)
            rm_linear_custom, rm_linear_custom_non_cum = read_rm(equation, "custom_loss", dp_type, training_distribution, model_type, distribution_models, distance)
            rm_RF, rm_RF_non_cum = read_rm(equation, loss, dp_type, training_distribution, "RF", distribution_models, distance)
            rm_LR,rm_LR_non_cum = read_rm(equation, loss, dp_type, training_distribution, "LR", distribution_models, distance)
            rm_cnn,rm_cnn_non_cum = read_rm(equation, loss, dp_type, training_distribution, "cnn", distribution_models, distance)
     
            mse_linear = read_mse(equation, loss, dp_type, training_distribution, model_type, distribution_models, distance)
            mse_linear_custom = read_mse(equation, "custom_loss", dp_type, training_distribution, model_type, distribution_models, distance)
            mse_RF = read_mse(equation, loss, dp_type, training_distribution, "RF", distribution_models, distance)
            mse_LR = read_mse(equation, loss, dp_type, training_distribution, "LR", distribution_models, distance)
            mse_cnn = read_mse(equation, loss, dp_type, training_distribution, "cnn", distribution_models, distance)
     
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
     
            
            plt.clf()
            fix, ax1 = plt.subplots()
            
            # ax1.plot(percentages, rm_linear, label="Linear Model", color="b")
            # ax1.plot(percentages, rm_linear_custom, label="Linear Model - MSE + Penalty", color="g")
            ax1.plot(percentages, rm_RF, label="RF", color="r")
            ax1.plot(percentages, rm_LR, label="LR", color="c")
            ax1.plot(percentages, rm_cnn, label="CNN", color="m")
            
            # plt.legend()
            ax1.set_xlabel("Percentage")
            ax1.set_ylabel("RM")
            ax1.set_title(f"RM vs MSE - {models_names_text[model_type]} Model - {training_distribution} - {distance} - {testing_distribution}")
            
            ax2= ax1.twinx()
            # ax2.plot(percentages, mse_linear, label="Linear Model", color="b", linestyle="--")
            # ax2.plot(percentages, mse_linear_custom, label="Linear Model - MSE + Penalty", color="g", linestyle="--")
            ax2.plot(percentages, mse_RF, label="RF", color="r", linestyle="--")
            ax2.plot(percentages, mse_LR, label="LR", color="c", linestyle="--")
            ax2.plot(percentages, mse_cnn, label="CNN", color="m", linestyle="--")
            
            ax2.set_ylabel("MSE")
            
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

            path = "./testing_distributions/"
            if os.path.exists(path) == False:
                os.makedirs(path)
            plt.savefig(f"./{path}/rm_vs_mse_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
            print("figures is saved in ", f"./{path}/rm_vs_mse_{model_type}_{training_distribution}_{distance}_models_{loss}_testing_dist_{testing_distribution}.png")
            
     
            
    # compare_models_mse_vs_rm(equation, loss, dp_type, training_distribution, model_type, normal_models_percentages, "Euclidean", "normal")
    # compare_models_mse_vs_rm(equation, loss, dp_type, training_distribution, model_type, uniform_models_percentages, "Euclidean", "uniform")
    # compare_models_mse_vs_rm(equation, loss, dp_type, training_distribution, model_type, laplace_models_percentages, "Euclidean", "laplace")
    
    def plot_mse(equation, loss, dp_type, training_distribution, model_type, models_set, distance, epsilon=None):
        variances = [0.1, 0.4, 0.6, 0.8, 1]
        mse = read_mse(equation, loss, dp_type, training_distribution, model_type, models_set, distance, epsilon)
        print("mse", mse)
        plt.clf()
        plt.plot(variances, mse)
        plt.xlabel("Noise Variance")
        plt.ylabel("MSE")
        plt.title(f"MSE vs Noise Variance - {models_names_text[model_type]} Model - {training_distribution} - {distance}")
        path = "./mse/"
        if os.path.exists(path) == False:
            os.makedirs(path)
        plt.savefig(f"./{path}//mse_vs_noise_variance_{model_type}_{training_distribution}_{distance}_models_{loss}.png")
        print("figures is saved in ", f"./{path}//mse_vs_noise_variance_{model_type}_{training_distribution}_{distance}_models_{loss}.png")
    # plot_mse(equation, loss, dp_type, training_distribution, model_type, normal_models, "Euclidean")
    # exit()
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #compare the percentages of normal distribution for different models 
    def compare_percentages(equation, loss, dp_type, training_distribution, model_type, normal_models):
        linear = read_rm(equation, loss, dp_type, training_distribution, model_type, normal_models, "Euclidean")
        linear_custom = read_rm(equation, "custom_loss", dp_type, training_distribution, model_type, normal_models, "Euclidean")
        rf = read_rm(equation, loss, dp_type, training_distribution, "RF", normal_models, "Euclidean")
        lr = read_rm(equation, loss, dp_type, training_distribution, "LR", normal_models, "Euclidean")
        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        
        plt.clf()
        plt.plot(percentages, linear, label="Linear Model")
        plt.plot(percentages, linear_custom, label="Linear Model - MSE + Penalty")
        plt.plot(percentages, rf, label="RF")
        plt.plot(percentages, lr, label="LR")
        plt.legend()
        plt.xlabel("Percentage")
        plt.ylabel("RM")
        plt.title(f"RM vs Percentage - {models_names_text[model_type]} Model - {training_distribution} - Euclidean")
        path = "./percentages/"
        if os.path.exists(path) == False:
            os.makedirs(path)
        plt.savefig(f"./{path}//rm_vs_percentage_{model_type}_{training_distribution}_models_{loss}.png")
    # normal_percentage_models = ["n28", "n29", "n30", "n31", "n32", "n33", "n34", "n35", "n36", "n37"]
    # compare_percentages(equation, loss, dp_type, training_distribution, model_type, normal_percentage_models)
    
    
    # exit()
        
    distance = "Euclidean"
    if training_distribution == "normal":
        training_epsilons = [0]
      
    for epsilon in training_epsilons:
        normal_rm_linear = read_rm(equation, loss, dp_type, training_distribution, model_type, normal_models, distance, epsilon)
        normal_rm_RF = read_rm(equation, loss, dp_type, training_distribution, "RF", normal_models, distance, epsilon)
        normal_rm_LR = read_rm(equation, loss, dp_type, training_distribution, "LR", normal_models, distance, epsilon)
        
        uniform_rm_linear = read_rm(equation, loss, dp_type, training_distribution, model_type, uniform_models, distance, epsilon)
        uniform_rm_RF = read_rm(equation, loss, dp_type, training_distribution, "RF", uniform_models, distance, epsilon)
        uniform_rm_LR = read_rm(equation, loss, dp_type, training_distribution, "LR", uniform_models, distance, epsilon)
        
        laplace_rm_linear = read_rm(equation, loss, dp_type, training_distribution, model_type, laplace_models,distance, epsilon)
        laplace_rm_RF = read_rm(equation, loss, dp_type, training_distribution, "RF", laplace_models,distance, epsilon)
        laplace_rm_LR = read_rm(equation, loss, dp_type, training_distribution, "LR", laplace_models,distance, epsilon)
        
        custom_loss_normal_rm_linear = read_rm(equation, "custom_loss", dp_type, training_distribution, model_type, normal_models, distance, epsilon)
        custom_loss_uniform_rm_linear = read_rm(equation, "custom_loss", dp_type, training_distribution, model_type, uniform_models, distance, epsilon)
        custom_loss_laplace_rm_linear = read_rm(equation, "custom_loss", dp_type, training_distribution, model_type, laplace_models,distance, epsilon)
        
        # compare linear model and different distributions
        plt.clf()
        # smaller font size for the legend
        plt.rcParams.update({'font.size': 10})
        plt.plot(variances, normal_rm_linear, label="Normal Distribution - MSE")
        plt.plot(variances, uniform_rm_linear, label="Uniform Distribution - MSE")
        plt.plot(variances, custom_loss_normal_rm_linear, label="Normal Distribution - MSE + penalty")
        plt.plot(variances, custom_loss_uniform_rm_linear, label="Uniform Distribution - MSE + penalty")
        plt.legend()
        plt.xlabel("Noise Variance")
        plt.ylabel("RM")
        plt.title("RM vs Noise Variance")
        plt.savefig(f"./rm_vs_noise_variance_linear_{distance}_training_dist_{training_distribution}_training_eps_{epsilon}.png")
        
        # compare models with normal distribution and different models
        plt.clf()
        plt.plot(variances, normal_rm_linear, label="Linear Model - MSE")
        plt.plot(variances, custom_loss_normal_rm_linear, label="Linear Model - MSE + penalty")
        plt.plot(variances, normal_rm_RF, label="RF")
        plt.plot(variances, normal_rm_LR, label="LR")
        plt.legend()
        plt.xlabel("Noise Variance")
        plt.ylabel("RM")
        plt.title("RM vs Noise Variance - Normal Distribution")
        plt.savefig(f"./rm_vs_noise_variance_models_normal_{distance}_training_dist_{training_distribution}_training_eps_{epsilon}.png")
        
        # compare models with uniform distribution and different models
        plt.clf()
        plt.plot(variances, uniform_rm_linear, label="Linear - MSE")
        plt.plot(variances, custom_loss_uniform_rm_linear, label="Linear - MSE + penalty")
        plt.plot(variances, uniform_rm_RF, label="RF")
        plt.plot(variances, uniform_rm_LR, label="LR")
        plt.legend()
        plt.xlabel("Noise Variance")
        plt.ylabel("RM")
        plt.title("RM vs Noise Variance - Uniform Distribution")
        plt.savefig(f"./rm_vs_noise_variance_models_uniform_{distance}_training_dist_{training_distribution}_training_eps_{epsilon}.png")
        
        # compare models with laplace distribution and different models
        plt.clf()
        plt.plot(epsilons, laplace_rm_linear, label="Linear - MSE")
        plt.plot(epsilons, custom_loss_laplace_rm_linear, label="Linear - MSE + Penalty")
        plt.plot(epsilons, laplace_rm_RF, label="RF")
        plt.plot(epsilons, laplace_rm_LR, label="LR")
        plt.legend()
        plt.xlabel("Noise Epsilon")
        plt.ylabel("RM")
        plt.title("RM vs Noise Epsilon - Laplace Distribution")
        plt.savefig(f"./rm_vs_noise_epsilon_models_laplace_{distance}_training_dist_{training_distribution}_training_eps_{epsilon}.png")