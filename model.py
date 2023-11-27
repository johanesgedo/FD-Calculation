import subprocess
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

def Compile_and_Run_CUDA_Code(Code_Name):
    subprocess.run(["nvcc", "-o", "CUDA_Kernel", "CUDA_Kernel.cu"])
    subprocess.run(["./CUDA_Kernel"])

def Load_Data(File_Names):
    data_list = []
    for File_Name in File_Names:
        with open(File_Name) as file:
            data = np.loadtxt(file)
        data_list.append(data)
    return data_list

def Exponential_Function(x, a, b, c):
    return a * np.exp(b * x) + c

def Logarithmic_Function(x, a, b, c):
    return np.log(a * x - c) / b

def Plot_Elapsed_Time_Data(x, data_list, labels=(), colors=(), linestyles=()):
    plt.figure(figsize=(10, 6))
    for i, dataset in enumerate(data_list):
        plt.plot(x, dataset, label=labels[i], color=colors[i], linestyle=linestyles[i])
    plt.title('Elapsed Time of Kernel Execution')
    plt.xlabel('Index of Elapsed Time')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.xlim(1, len(x)-1) # Set x-axis limit
    plt.xscale('log')     # Set x-axis to log scale
    plt.yscale('log')     # Set y-axis to log scale
    plt.show(block=False)

def Exponential_Model_of_Elapsed_Time(x_values, data_list=None, file_paths=None, labels=(), colors=(), linestyles=(), fit_curve=True):
    plt.figure(figsize=(10, 6))
    if data_list is not None:
        # Case 1: Data have been loaded
        for i, dataset in enumerate(data_list):
            popt, _ = curve_fit(Exponential_Function, x_values, dataset)
            y_fit = Exponential_Function(x_values, *popt)
            plt.plot(x_values, dataset, label=labels[i], color=colors[i], linestyle=linestyles[i], marker='o')
            if fit_curve:
                plt.plot(x_values, y_fit, linestyle='-', color='red')
    elif file_paths is not None:
        # Case 2: Load data from file_paths
        for i, file_path in enumerate(file_paths):
            data = Load_Data([file_path])
            popt, _ = curve_fit(Exponential_Function, x_values, data[0])
            y_fit = Exponential_Function(x_values, *popt)
            plt.plot(x_values, data[0], label=labels[i], color=colors[i], linestyle=linestyles[i], marker='o')
            if fit_curve:
                plt.plot(x_values, y_fit, linestyle='-', color='red')
    plt.title('Exponential Model of Elapsed Time')
    plt.xlabel('Index of Elapsed Time in Symetric Logaritmic Scale')
    plt.ylabel('Time (ms) in Symetric Logarithmic Scale')
    plt.legend()
    plt.grid(True)
    plt.xlim(1, len(x_values) - 1)  # Set x-axis limit
    plt.xscale('symlog')  # Set x-axis to symetric logarithmic scale
    plt.yscale('symlog')  # Set y-axis to symetric logarithmic scale
    plt.show(block=False)

def Logarithmic_Model_of_Elapsed_Time(x_values, data_list=None, file_paths=None, labels=(), colors=(), linestyles=(), fit_curve=True):
    plt.figure(figsize=(10, 6))
    if data_list is not None:
        # Case 1: Data have been loaded
        for i, dataset in enumerate(data_list):
            popt, _ = curve_fit(Logarithmic_Function, x_values, dataset)
            y_fit = Logarithmic_Function(x_values, *popt)
            plt.plot(x_values, dataset, label=labels[i], color=colors[i], linestyle=linestyles[i], marker='o')
            if fit_curve:
                plt.plot(x_values, y_fit, linestyle='-', color='red')
    elif file_paths is not None:
        # Case 2: Load data from file_paths
        for i, file_path in enumerate(file_paths):
            data = Load_Data([file_path])
            popt, _ = curve_fit(Logarithmic_Function, x_values, data[0])
            y_fit = Logarithmic_Function(x_values, *popt)
            plt.plot(x_values, data[0], label=labels[i], color=colors[i], linestyle=linestyles[i], marker='o')
            if fit_curve:
                plt.plot(x_values, y_fit, linestyle='-', color='red')
    plt.title('Logarithmic Model of Elapsed Time')
    plt.xlabel('Index of Elapsed Time in Logarithmic Scale')
    plt.ylabel('Time (ms) in Logarithmic Scale')
    plt.legend()
    plt.grid(True)
    plt.xlim(1, len(x_values) - 1)  # Set x-axis limit
    plt.xscale('log')  # Set x-axis to log scale
    plt.yscale('log')  # Set y-axis to log scale
    plt.show()

def Plot_Result_Data(x, data_list, labels=(), colors=(), linestyles=()):
    plt.figure(figsize=(10, 6))
    max_points = 1000
    interval = max(1, len(x) // max_points)
    for i, dataset in enumerate(data_list):
        plt.plot(x[:max_points:interval], dataset[:max_points:interval], label=labels[i], color=colors[i], linestyle=linestyles[i])
    plt.title('Calculation Results')
    plt.xlabel('Index Data')
    plt.ylabel('Data')
    plt.legend()
    plt.grid(True)
    plt.xlim(1, len(x)-1) # Set x-axis limit
    plt.xscale('log')     # Set x-axis to log scale
    plt.yscale('log')     # Set y-axis to log scale
    plt.show()


def main():
    Code_Name = "CUDA_Kernel"
    Compile_and_Run_CUDA_Code(Code_Name)

    # List of Elapsed file names
    Elapsed_File_Names = ["GalerkinTime_data.txt",
                          "LeapfrogTime_data.txt",
                          "CrankNicolsonTime_data.txt",
                          "ADITime_data.txt",
                          "SigmaTime_data.txt",
                          "LaxWendroffTime_data.txt",
                          "FractionalStepTime_data.txt",
                          "MacCormackTime_data.txt",
                          "TVDTime_data.txt",
                          "PSORTime_data.txt",
                          "FVSTime_data.txt"]

    file_paths = ["GalerkinTime_data.txt",
                  "LeapfrogTime_data.txt",
                  "CrankNicolsonTime_data.txt",
                  "ADITime_data.txt",
                  "SigmaTime_data.txt",
                  "LaxWendroffTime_data.txt",
                  "FractionalStepTime_data.txt",
                  "MacCormackTime_data.txt",
                  "TVDTime_data.txt",
                  "PSORTime_data.txt",
                  "FVSTime_data.txt"]

 #   Result_File_Names = ["GalerkinSolver.txt",
 #                        "LeapfrogSolver.txt",
 #                        "CrankNicolsonSolver.txt",
 #                        "ADISolver.txt",
 #                        "SigmaSolver.txt",
 #                        "LaxWendroffSolver.txt",
 #                        "FractionalStepSolver.txt",
 #                        "MacCormackSolver.txt",
 #                        "TVDSolver.txt",
 #                        "PSORSolver.txt",
 #                        "FVSSolver.txt"]

    # List to store data
    Elapsed_Data_List = Load_Data(Elapsed_File_Names)
#    Result_Data_List = [Load_Data(File_Name) for File_Name in Result_File_Names]

    # Plot Elapsed Time Data
    x_values = np.arange(1, len(Elapsed_Data_List[0]) + 1)

    Plot_Elapsed_Time_Data(x_values, Elapsed_Data_List,
                           labels=['Galerkin', 'Leapfrog', 'CrankNicolson', 'ADI', 'Sigma', 'LaxWendroff',
                                   'FractionalStep', 'MacCormack', 'TVD', 'PSOR', 'FVS'],
                           colors=['blue','red', 'green', 'orange', 'purple', 'yellow',
                                   'magenta', 'brown', 'black', 'cyan', 'gray'],
                           linestyles=['-', '--', '-.', 'dashed', ':', 'dotted',
                                       'solid', 'dashed', 'dashdot', 'dotted', ':'])

    Exponential_Model_of_Elapsed_Time(x_values, data_list=Elapsed_Data_List,
                           labels=['Galerkin', 'Leapfrog', 'CrankNicolson', 'ADI', 'Sigma', 'LaxWendroff',
                                   'FractionalStep', 'MacCormack', 'TVD', 'PSOR', 'FVS'],
                           colors=['blue','red', 'green', 'orange', 'purple', 'yellow',
                                   'magenta', 'brown', 'black', 'cyan', 'gray'],
                           linestyles=['-', '--', '-.', 'dashed', ':', 'dotted',
                                       'solid', 'dashed', 'dashdot', 'dotted', ':'],
                            fit_curve=True)

    Exponential_Model_of_Elapsed_Time(x_values, file_paths=file_paths,
                           labels=['Galerkin', 'Leapfrog', 'CrankNicolson', 'ADI', 'Sigma', 'LaxWendroff',
                                   'FractionalStep', 'MacCormack', 'TVD', 'PSOR', 'FVS'],
                           colors=['blue','red', 'green', 'orange', 'purple', 'yellow',
                                   'magenta', 'brown', 'black', 'cyan', 'gray'],
                           linestyles=['-', '--', '-.', 'dashed', ':', 'dotted',
                                       'solid', 'dashed', 'dashdot', 'dotted', ':'],
                            fit_curve=True)

    Logarithmic_Model_of_Elapsed_Time(x_values, data_list=Elapsed_Data_List,
                           labels=['Galerkin', 'Leapfrog', 'CrankNicolson', 'ADI', 'Sigma', 'LaxWendroff',
                                   'FractionalStep', 'MacCormack', 'TVD', 'PSOR', 'FVS'],
                           colors=['blue','red', 'green', 'orange', 'purple', 'yellow',
                                   'magenta', 'brown', 'black', 'cyan', 'gray'],
                           linestyles=['-', '--', '-.', 'dashed', ':', 'dotted',
                                       'solid', 'dashed', 'dashdot', 'dotted', ':'],
                            fit_curve=True)

    Logarithmic_Model_of_Elapsed_Time(x_values, file_paths=file_paths,
                           labels=['Galerkin', 'Leapfrog', 'CrankNicolson', 'ADI', 'Sigma', 'LaxWendroff',
                                   'FractionalStep', 'MacCormack', 'TVD', 'PSOR', 'FVS'],
                           colors=['blue','red', 'green', 'orange', 'purple', 'yellow',
                                   'magenta', 'brown', 'black', 'cyan', 'gray'],
                           linestyles=['-', '--', '-.', 'dashed', ':', 'dotted',
                                       'solid', 'dashed', 'dashdot', 'dotted', ':'],
                            fit_curve=True)
    
    # Plot Result Data
 #   Plot_Result_Data(np.arange(1, len(Result_Data_List[0]) + 1), Result_Data_List,
 #                          labels=['Galerkin', 'Leapfrog', 'TVD'],
 #                          colors=['blue', 'red', 'green'],
 #                          linestyles=['-', '--', '-.', '-..', ':', '4',
 #                                      '-', '--', '-.', '-..', ':'])
    #plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    main()

