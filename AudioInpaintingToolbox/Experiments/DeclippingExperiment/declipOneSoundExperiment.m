function declipOneSoundExperiment(expParam)
% A simple experiment to declip a signal.
%
% Usage: declipOneSoundExperiment(expParam)
%
%
% Inputs:
%          - expParam is an optional structure where the user can define
%          the experiment parameters.
%          - expParam.clippingLevel: clipping level between 0 and 1.
%          - expParam.filename: file to be tested.
%          - expParam.destDir: path to store the results.
%          - expParam.solver: solver with its parameters
%          - expParam.destDir: path to store the results.
%
%
% -------------------
%
% Audio Inpainting toolbox
% Date: June 28, 2011
% By Valentin Emiya, Amir Adler, Maria Jafari
% This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).

if ~isdeployed
    close all
    addpath('../../Problems/');
    addpath('../../Solvers/');
    addpath('../../Utils/');
    addpath('../../Utils/dictionaries/');
    addpath('../../Utils/evaluation/');
%     addpath('../../Utils/TCPIP_SocketCom/');
%     javaaddpath('../../Utils/TCPIP_SocketCom');
    dbstop if error
end

%% Set parameters
if nargin<1
    expParam = [];
end
if ~isfield(expParam,'filename')
    expParam.filename = '../../Data/testSpeech8kHz_from16kHz/male01_8kHz.wav';
end
if ~isfield(expParam,'clippingLevel')
    expParam.clippingLevel = 0.6;
end

% Solver
if ~isfield(expParam,'solver')
    warning('AITB:N','Frame length=256 is used to have faster computations. Recommended frame length is 512 at 8kHz.');
    warning('AITB:overlap','Overlap factor=2 is used to have faster computations. Recommended value: 4.');
    
    expParam.solver.name = 'OMP-G';
    expParam.solver.function = @inpaintSignal_IndependentProcessingOfFrames;
    expParam.solver.param.N = 512; % frame length
    expParam.solver.param.N = 256; % frame length
    expParam.solver.param.inpaintFrame = @inpaintFrame_OMP_Gabor; % solver function
    expParam.solver.param.OMPerr = 0.001;
    expParam.solver.param.sparsityDegree = expParam.solver.param.N/4;
    expParam.solver.param.D_fun = @Gabor_Dictionary; % Dictionary (function handle)
    expParam.solver.param.OLA_frameOverlapFactor = 4;
    expParam.solver.param.OLA_frameOverlapFactor = 2;
    expParam.solver.param.redundancyFactor = 2; % Dictionary redundancy
    expParam.solver.param.wd = @wRect; % Weighting window for dictionary atoms
    expParam.solver.param.wa = @wRect; % Analysis window
    expParam.solver.param.OLA_ws = @wSine; % Synthesis window
    expParam.solver.param.SKIP_CLEAN_FRAMES = true; % do not process frames where there is no missing samples
    expParam.solver.param.MULTITHREAD_FRAME_PROCESSING = false; % not implemented yet
end
if ~isfield(expParam,'destDir'),
    expParam.destDir = '../../tmp/declipOneSound/';
end
if ~exist(expParam.destDir,'dir')
    mkdir(expParam.destDir)
end

%% Read test signal
[x fs] = wavread(expParam.filename);

%% Generate the problem
[problemData, solutionData] = generateDeclippingProblem(x,expParam.clippingLevel);

%% Declip with solver
fprintf('\nDeclipping\n')
% [xEst1 xEst2] = inpaintSignal_IndependentProcessingOfFrames(problemData,param);
solverParam = expParam.solver.param;
[xEst1 xEst2] = expParam.solver.function(problemData,solverParam);

%% Compute and display SNR performance
L = length(xEst1);
N = expParam.solver.param.N;
[SNRAll, SNRmiss] = SNRInpaintingPerformance(...
    solutionData.xClean(N:L-N),problemData.x(N:L-N),...
    xEst2(N:L-N),problemData.IMiss(N:L-N));
fprintf('SNR on missing samples:\n');
fprintf('Clipped: %g dB\n',SNRmiss(1));
fprintf('Estimate: %g dB\n',SNRmiss(2));


% Plot results
xClipped = problemData.x;
xClean = solutionData.xClean;
figure
hold on
plot(xClipped,'r')
plot(xClean)
plot(xEst2,'--g')
plot([1;length(xClipped)],[1;1]*[-1,1]*max(abs(xClipped)),':r')
legend('Clipped','True solution','Estimate')

% Normalized and save sounds
normX = 1.1*max(abs([xEst1(:);xEst2(:);xClean(:)]));
L = min([length(xEst2),length(xEst1),length(xClean),length(xEst1),length(xClipped)]);
xEst1 = xEst1(1:L)/normX;
xEst2 = xEst2(1:L)/normX;
xClipped = xClipped(1:L)/normX;
xClean = xClean(1:L)/normX;
% FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/DeclippingExperiment/declipOneSoundExperiment.m
% BEGIN: ed8c6549bwf9
audiowrite([expParam.destDir 'xEst1.wav'], xEst1, fs);
% END: ed8c6549bwf9
% FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/DeclippingExperiment/declipOneSoundExperiment.m
% BEGIN: ed8c6549bwf9
audiowrite([expParam.destDir 'xEst2.wav'], xEst2, fs);
% END: ed8c6549bwf9
wavwrite(xClipped,fs,[expParam.destDir 'xClipped.wav']);
wavwrite(xClean,fs,[expParam.destDir 'xClean.wav']);


return
