
-------------------------------------------------

Audio Inpainting Toolbox

By

Valentin Emiya, INRIA, France
Amir Adler, The Technion, Israel
Maria Jafari, Queen Mary University of London, UK

Contact: valentin.emiya@inria.fr
-------------------------------------------------

%%%%%%%%%%%%
Requirements
%%%%%%%%%%%%
The code has been developped in Matlab (R2010a).
The CVX toolbox is required by solvers.

%%%%%%%%%%%%
Installation
%%%%%%%%%%%%
Just unpack the archive and ensure that you have CVX installed.

%%%%%%%%%%%%%%%
Getting started
%%%%%%%%%%%%%%%
Just run the files in the subdirectories of 'Experiments/'.
As a starting example, the simplest one is declipOneSoundExperiment.m.

%%%%%%%%%%%%%%%%%%%
Very quick tutorial
%%%%%%%%%%%%%%%%%%%
The toolbox is organized into several types of components, each type being located in a separate directory:
- Problems (API: '[problemData,solutionData] = generateMyProblem(mysound,problemParam);'): generates a particular problem (e.g. "declip this sound"), with given parameters, and generates the true solution.
- Solvers/algorithms (API: 'solutionEstimate = mySolver(problemData,solverParameters);'): given a problem and the solver parameters (a dictionary, thresholds, and so on), a solver proposes a solution using its particular algorithm
- Utils: e.g. dictionaries, evaluation functions are stored here
- Data: audio datasets including speech, music
- Experiments (API: 'myExperiment(experimentParameters);'): they are the main files one may run. A specific experiment takes a dataset, generates specific problems (e.g. increasing clipping levels), solves each problem with a number of solvers (specified in the experiment parameters), displays the performance for each solver. The experiments can be run without any input argument. In this case, default values will be used.

You may find more information:
- about each function 'myFunction', by typing 'help myFunction' in Matlab
- in the documented code of each function
- in the extended abstract and slides presented at the SPARS'11 workshop
- in the paper available at http://hal.inria.fr/inria-00577079/en

%%%%%%%%%%%%%%%%%%%%%%%%
How to cite this toolbox
%%%%%%%%%%%%%%%%%%%%%%%%
Please cite the following paper:
Adler Amir; Emiya Valentin; Jafari Maria; Elad Michael; Gribonval Remi; Plumbley Mark
Audio Inpainting
Submitted to IEEE Transactions on Audio, Speech, and Language Processing (2011)
Available at http://hal.inria.fr/inria-00577079/en.

%%%%%%%%%%%%%%%%%%%%%%%%%%%
Known issues / Future works
%%%%%%%%%%%%%%%%%%%%%%%%%%%
- The multithread processing of audio frames is not yet available (you may wonder about the Java TCP/IP utils, which will be used soon for this purpose).
- Some solvers based on L1 minimization will be added soon.
- The experiment called 'FromSmallToLargeHoleExperiment' will be added soon.

%%%%%%%
License
%%%%%%%
The code of this toolbox is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).
The data files are distributed under specific licenses as stated in the related .txt files in the directory 'Data/'.


