Leah Bar, September 2020
barleah.libra@gmail.com



Requires: Matlab
Python 3.x
Tensorflow 1.14


EIT Forward problem
================================================
Let us denote the project directory "root".

1) We first prepare our data. In the code we pick a phantom (1 or 2), calculate the ground truth solution by the FEM method, 
save the solution in a 1000x1000 grid, prepare the boundary conditions.
n is the frequency of the electrical current and phs [0,7] is the phase such that phase =phs*pi/8.
In the example we took phantom=2, n=1, phs=0.

% cd root/Matlab
% PrepareData.m
Then a mat file (Grid1kL5H2n1_phase_0.mat) is written in root/Data/InputData


2) Solution of the forward problem by a neural network
$ cd root/Python
$ python3 mainSearchU.py

The output (Grid1kL5H2n1_phase_0_output.mat, Grid1kL5H2n1_phase_0_output.png) is written in root/Data/OutputData.


3) Calculate error metrics in
% cd root/Matlab
% DisplayResultsU.m

if writeOutput = 1, (line 7)  figures are saved in root/Data/OutputData/Figures.




EIT Semi-Inverse Problem
===========================================================================

4) Produce 4 different forward problems (step 1) using root/Matlab/PrepareData.m e.g.
n=1 phs=0, n=1 phs=4, n=2 phs=0, n=2 phs=4
mat files are written in root/Data/InputData
Grid1kL5H2n1_phase_0.mat
Grid1kL5H2n1_phase_4.mat
Grid1kL5H2n2_phase_0.mat
Grid1kL5H2n2_phase_4.mat



5) Pack the 4 measurements into one file
$cd root/Matlab
$ PackU.m
the file is: root/Data/OutputData/Uest1kL5H2all.mat

6) Calculate sigma by the neural network
$ cd root/Python
$ python3 mainSearchSigma.py
Reconstructed sigma is saved in root/Data/OutputData as an image and mat file (Sest1kL5H2_1234.mat, Sest1kL5H2_1234.png).

7) Display sigma with error calculation
% cd root/Matlab
% DisplayResultsS 
if writeOutput = 1, (line 7)  figures are saved in root/Data/OutputData/Figures.
