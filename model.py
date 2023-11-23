import subprocess
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def Compile_and_Run_CUDA_Code(Code_Name):
    subprocess.run(["nvcc", "-o", "CUDA_Kernel", "CUDA_Kernel.cu"])
    subprocess.run(["./CUDA_Kernel"])

def Load_Data(File_Name):
    with open(File_Name) as file:
        data = np.loadtxt(file)
    return data

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
    Elapsed_File_Names = ["GalerkinTime_data.txt", "LeapfrogTime_data.txt", "TVDTime_data.txt"]
    Result_File_Names = ["GalerkinSolver.txt", "LeapfrogSolver.txt", "TVDSolver.txt"]

    # List to store data
    Elapsed_Data_List = [Load_Data(File_Name) for File_Name in Elapsed_File_Names]
    Result_Data_List = [Load_Data(File_Name) for File_Name in Result_File_Names]

    # Plot Elapsed Time Data
    Plot_Elapsed_Time_Data(np.arange(1, len(Elapsed_Data_List[0]) + 1), Elapsed_Data_List,
                           labels=['Galerkin', 'Leapfrog', 'TVD'],
                           colors=['blue', 'red', 'green'],
                           linestyles=['-', '--', '-.'])
    # Plot Result Data
    Plot_Result_Data(np.arange(1, len(Result_Data_List[0]) + 1), Result_Data_List,
                           labels=['Galerkin', 'Leapfrog', 'TVD'],
                           colors=['blue', 'red', 'green'],
                           linestyles=['-', '--', '-.'])
    #plt.tight_layout()
    #plt.show()

if __name__ == "__main__":
    main()

