# 3D-FD-Calculation
Finite difference calculation to solve 3D partial differential equation in space-time domain

I run these codes on WSL Ubuntu 22.04 with CUDA and Anaconda environments.
Instruction for using the codes:
1. Make sure the Ubuntu or Linux using has CUDA toolkits and Anaconda installed
2. type "python model.py" in the Linux terminal to execute the program
3. The file "CUDA_Kernel.cu" will be automatically installed with nvcc when running the command above.
4. If an error occurs, then some of the libraries applied to the program have not been installed properly.
5. Run the "clean.sh" by typing "sh clean.sh" in the Linux terminal to clean the data.
