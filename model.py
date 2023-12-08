import subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def Compile_and_Run_CUDA_Code(input_file, output_name):
    subprocess.run(["nvcc", "-o", output_name, input_file])
    subprocess.run(["./" + output_name])

def Load_Data(File_Names):
    data_list = []
    for File_Name in File_Names:
        with open(File_Name) as file:
            data = np.loadtxt(file)
        data_list.append(data)
    return data_list

def Generate_File_Names(Dimension):
    methods = ["Galerkin", "Leapfrog", "CrankNicolson", "ADI", "Sigma", "LaxWendroff",
               "FractionalStep", "MacCormack", "TVD", "PSOR", "FVS"]
    return [f"{method}Time_{Dimension}_data.txt" for method in methods]

def Exponential_Function(x, a, b, c):
    return np.where(x >= 0, a * np.exp(b * x) + c, np.nan)

def Logarithmic_Function(x, a, b, c):
    valid_indices = x > 0  # Filter out non-positive values
    return np.where(valid_indices, np.log(a * x[valid_indices] - c) / b, np.nan)

def Model_of_Elapsed_Time(x_values, data_list=None, file_paths=None, fit_function=None, labels=(), colors=(), linestyles=(), fit_curve=True, plot_label=None):
    plt.figure(figsize=(10, 6))
    if data_list is not None:
        # Case 1: Data have been loaded
        for i, dataset in enumerate(data_list):
            popt, _ = curve_fit(fit_function, x_values, dataset)
            y_fit = fit_function(x_values, *popt)
            plt.plot(x_values, dataset, label=labels[i], color=colors[i], linestyle=linestyles[i], marker='o')
            if fit_curve:
                plt.plot(x_values, y_fit, linestyle='-', color='red')
    elif file_paths is not None:
        # Case 2: Load data from file_paths
        for i, file_path in enumerate(file_paths):
            data = Load_Data([file_path])
            popt, _ = curve_fit(fit_function, x_values, data[0])
            y_fit = fit_function(x_values, *popt)
            plt.plot(x_values, data[0], label=labels[i], color=colors[i], linestyle=linestyles[i], marker='o')
            if fit_curve:
                plt.plot(x_values, y_fit, linestyle='-', color='red')
#    if plot_label is not None:
#        plt.text(0.5, 0.95, plot_label, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    if plot_label is not None:
        plt.title(f'{fit_function.__name__} Model of {plot_label} Elapsed Time')
    else:
        plt.title(f'{fit_function.__name__} Model of Elapsed Time')
    plt.xlabel(f'Index of Elapsed Time in {"Symetric Logaritmic" if fit_function == Exponential_Function else "Logarithmic"} Scale')
    plt.ylabel(f'Time (ms) in {"Symetric Logarithmic" if fit_function == Exponential_Function else "Logarithmic"} Scale')
    plt.legend()
    plt.grid(True)
    plt.xlim(1, len(x_values) - 1)  # Set x-axis limit
    plt.xscale('symlog' if fit_function == Exponential_Function else 'log')
    plt.yscale('symlog' if fit_function == Exponential_Function else 'log')
    plt.show()

def main():
    Compile_and_Run_CUDA_Code("CUDA_3DKernel.cu", "CUDA_3DKernel")
    Compile_and_Run_CUDA_Code("CUDA_2DKernel.cu", "CUDA_2DKernel")

    # List of Elapsed file names
    elapsed_2d_file_names = Generate_File_Names("2D")
    elapsed_3d_file_names = Generate_File_Names("3D")

    file_2d_paths = Generate_File_Names("2D")
    file_3d_paths = Generate_File_Names("3D")

    # List to store data
    elapsed_2d_data_list = Load_Data(elapsed_2d_file_names)
    elapsed_3d_data_list = Load_Data(elapsed_3d_file_names)

    Elapsed_Data_args = {
        'labels' : ['Galerkin', 'Leapfrog', 'CrankNicolson', 'ADI', 'Sigma', 'LaxWendroff',
                    'FractionalStep', 'MacCormack', 'TVD', 'PSOR', 'FVS'],
        'colors' : ['blue', 'red', 'green', 'orange', 'purple', 'yellow',
                    'magenta', 'brown', 'black', 'cyan', 'gray'],
        'linestyles': ['-', '--', '-.', 'dashed', ':', 'dotted',
                       'solid', 'dashed', 'dashdot', 'dotted', ':'],
        'fit_curve' : True
    }

    # Plot Elapsed Time Data and Models for 3D
    x_values_3d = np.arange(1, len(elapsed_3d_data_list[0]) + 1)
    Model_of_Elapsed_Time(x_values_3d, data_list=elapsed_3d_data_list, fit_function=Exponential_Function, plot_label="3D", **Elapsed_Data_args)
    Model_of_Elapsed_Time(x_values_3d, file_paths=file_3d_paths, fit_function=Logarithmic_Function, plot_label="3D", **Elapsed_Data_args)

    # Plot Elapsed Time Data and Models for 2D
    x_values_2d = np.arange(1, len(elapsed_2d_data_list[0]) + 1)
    Model_of_Elapsed_Time(x_values_2d, data_list=elapsed_2d_data_list, fit_function=Exponential_Function, plot_label="2D", **Elapsed_Data_args)
    Model_of_Elapsed_Time(x_values_2d, file_paths=file_2d_paths, fit_function=Logarithmic_Function, plot_label="2D", **Elapsed_Data_args)

if __name__ == "__main__":
    main()

