function declippingExperiment(expParam)
    % Declip several sounds with different clipping levels, using several
    % solvers.
    %
    % Usage: declippingExperiment(expParam)
    %
    %
    % Inputs:
    %          - expParam is an optional structure where the user can define
    %          the experiment parameters.
    %          - expParam.clippingScale: clipping values to test, as a vector
    %           of real numbers in ]0,1[.
    %          - expParam.soundDir: path to sound directory. All the .wav files
    %          in this directory will be tested.
    %          - expParam.destDir: path to store the results.
    %          - expParam.solvers: list of solvers with their parameters
    %
    %
    % -------------------
    %
    % Audio Inpainting toolbox
    % Date: June 28, 2011
    % By Valentin Emiya, Amir Adler, Maria Jafari
    % This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).
    
    if ~isdeployed
        addpath('../../Problems/');
        addpath('../../Solvers/');
        addpath('../../Utils/');
        addpath('../../Utils/dictionaries/');
        addpath('../../Utils/evaluation/');
    %     addpath('../../Utils/TCPIP_SocketCom/');
    %     javaaddpath('../../Utils/TCPIP_SocketCom');
        dbstop if error
        close all
    end
    
    if nargin<1
        expParam = [];
    end
    if ~isfield(expParam,'clippingScale')
        expParam.clippingScale = 0.4:0.2:0.8;
    end
    if ~isfield(expParam,'soundDir')
        expParam.soundDir = '../../Data/testSpeech8kHz_from16kHz/';
        expParam.soundDir = '../../Data/shortTest/';
        warning('AITB:soundDir','soundDir has only one sound to have faster computations. Recommended soundDir: ../../Data/testSpeech8kHz_from16kHz/');
    end
    if ~isfield(expParam,'destDir')
        expParam.destDir = '../../tmp/declip/';
    end
    
    %% Set parameters
    
    if ~isfield(expParam,'solvers'),
        % Choose the solver methods you would like to test: OMP, L1, Janssen
        warning('AITB:N','Frame length=256 is used to have faster computations. Recommended frame length is 512 at 8kHz.');
        warning('AITB:overlap','Overlap factor=2 is used to have faster computations. Recommended value: 4.');
        nSolver = 0;
        
        nSolver = nSolver+1;
        expParam.solvers(nSolver).name = 'OMP-C';
        expParam.solvers(nSolver).function = @inpaintSignal_IndependentProcessingOfFrames;
        expParam.solvers(nSolver).param.N = 512; % frame length
        expParam.solvers(nSolver).param.N = 256; % frame length
        expParam.solvers(nSolver).param.inpaintFrame = @inpaintFrame_OMP; % solver function
        expParam.solvers(nSolver).param.OMPerr = 0.001;
        expParam.solvers(nSolver).param.sparsityDegree = expParam.solvers(nSolver).param.N/4;
        expParam.solvers(nSolver).param.D_fun = @DCT_Dictionary; % Dictionary (function handle)
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 4;
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 2;
        expParam.solvers(nSolver).param.redundancyFactor = 2; % Dictionary redundancy
        expParam.solvers(nSolver).param.wd = @wRect; % Weighting window for dictionary atoms
        expParam.solvers(nSolver).param.wa = @wRect; % Analysis window
        expParam.solvers(nSolver).param.OLA_ws = @wSine; % Synthesis window
        expParam.solvers(nSolver).param.SKIP_CLEAN_FRAMES = true; % do not process frames where there is no missing samples
        expParam.solvers(nSolver).param.MULTITHREAD_FRAME_PROCESSING = false; % not implemented yet
        
        nSolver = nSolver+1;
        expParam.solvers(nSolver).name = 'consOMP-C';
        expParam.solvers(nSolver).function = @inpaintSignal_IndependentProcessingOfFrames;
        expParam.solvers(nSolver).param.N = 512; % frame length
        expParam.solvers(nSolver).param.N = 256; % frame length
        expParam.solvers(nSolver).param.inpaintFrame = @inpaintFrame_consOMP; % solver function
        expParam.solvers(nSolver).param.OMPerr = 0.001;
        expParam.solvers(nSolver).param.sparsityDegree = expParam.solvers(nSolver).param.N/4;
        expParam.solvers(nSolver).param.D_fun = @DCT_Dictionary; % Dictionary (function handle)
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 4;
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 2;
        expParam.solvers(nSolver).param.redundancyFactor = 2; % Dictionary redundancy
        expParam.solvers(nSolver).param.wd = @wRect; % Weighting window for dictionary atoms
        expParam.solvers(nSolver).param.wa = @wRect; % Analysis window
        expParam.solvers(nSolver).param.OLA_ws = @wSine; % Synthesis window
        expParam.solvers(nSolver).param.SKIP_CLEAN_FRAMES = true; % do not process frames where there is no missing samples
        expParam.solvers(nSolver).param.MULTITHREAD_FRAME_PROCESSING = false; % not implemented yet
        
        nSolver = nSolver+1;
        expParam.solvers(nSolver).name = 'OMP-G';
        expParam.solvers(nSolver).function = @inpaintSignal_IndependentProcessingOfFrames;
        expParam.solvers(nSolver).param.N = 512; % frame length
        expParam.solvers(nSolver).param.N = 256; % frame length
        expParam.solvers(nSolver).param.inpaintFrame = @inpaintFrame_OMP_Gabor; % solver function
        expParam.solvers(nSolver).param.OMPerr = 0.001;
        expParam.solvers(nSolver).param.sparsityDegree = expParam.solvers(nSolver).param.N/4;
        expParam.solvers(nSolver).param.D_fun = @Gabor_Dictionary; % Dictionary (function handle)
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 4;
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 2;
        expParam.solvers(nSolver).param.redundancyFactor = 2; % Dictionary redundancy
        expParam.solvers(nSolver).param.wd = @wRect; % Weighting window for dictionary atoms
        expParam.solvers(nSolver).param.wa = @wRect; % Analysis window
        expParam.solvers(nSolver).param.OLA_ws = @wSine; % Synthesis window
        expParam.solvers(nSolver).param.SKIP_CLEAN_FRAMES = true; % do not process frames where there is no missing samples
        expParam.solvers(nSolver).param.MULTITHREAD_FRAME_PROCESSING = false; % not implemented yet
        
        nSolver = nSolver+1;
        expParam.solvers(nSolver).name = 'consOMP-G';
        expParam.solvers(nSolver).function = @inpaintSignal_IndependentProcessingOfFrames;
        expParam.solvers(nSolver).param.N = 512; % frame length
        expParam.solvers(nSolver).param.N = 256; % frame length
        expParam.solvers(nSolver).param.inpaintFrame = @inpaintFrame_consOMP_Gabor; % solver function
        expParam.solvers(nSolver).param.OMPerr = 0.001;
        expParam.solvers(nSolver).param.sparsityDegree = expParam.solvers(nSolver).param.N/4;
        expParam.solvers(nSolver).param.D_fun = @Gabor_Dictionary; % Dictionary (function handle)
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 4;
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 2
        expParam.solvers(nSolver).param.redundancyFactor = 2; % Dictionary redundancy
        expParam.solvers(nSolver).param.wd = @wRect; % Weighting window for dictionary atoms
        expParam.solvers(nSolver).param.wa = @wRect; % Analysis window
        expParam.solvers(nSolver).param.OLA_ws = @wSine; % Synthesis window
        expParam.solvers(nSolver).param.SKIP_CLEAN_FRAMES = true; % do not process frames where there is no missing samples
        expParam.solvers(nSolver).param.MULTITHREAD_FRAME_PROCESSING = false; % not implemented yet
        
        nSolver = nSolver+1;
        expParam.solvers(nSolver).name = 'Janssen';
        expParam.solvers(nSolver).function = @inpaintSignal_IndependentProcessingOfFrames;
        expParam.solvers(nSolver).param.inpaintFrame = @inpaintFrame_janssenInterpolation; % solver function
        expParam.solvers(nSolver).param.N = 512; % frame length
        expParam.solvers(nSolver).param.N = 256;
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 4;
        expParam.solvers(nSolver).param.OLA_frameOverlapFactor = 2
        expParam.solvers(nSolver).param.wa = @wRect; % Analysis window
        expParam.solvers(nSolver).param.OLA_ws = @wSine; % Synthesis window
        expParam.solvers(nSolver).param.SKIP_CLEAN_FRAMES = true; % do not process frames where there is no missing samples
        expParam.solvers(nSolver).param.MULTITHREAD_FRAME_PROCESSING = false; % not implemented yet
    end
    
    SNRClip = zeros(0,0,0);
    fprintf('Folder %s\n',expParam.soundDir);
    if ~exist(expParam.destDir,'dir')
        mkdir(expParam.destDir)
    end
    soundFiles = dir([expParam.soundDir '*.wav']);
    
    for kf = 1:length(soundFiles)
        soundfile = [expParam.soundDir soundFiles(kf).name];
        fprintf(' File %s\n',soundfile);
        %% Read test signal
        [x, fs] = audioread(soundfile);
        
        for kClip = 1:length(expParam.clippingScale)
            clippingLevel = expParam.clippingScale(kClip);
            fprintf('  Clip level %g\n',clippingLevel);
            
            %% Generate the problem
            [problemData, solutionData] = generateDeclippingProblem(x,clippingLevel);
            
            for nSolver = 1:length(expParam.solvers)
                %% Declip with solver
                solverParam = expParam.solvers(nSolver).param;
                [xEst1, xEst2] = expParam.solvers(nSolver).function(problemData,solverParam);
                
                %% compute performance
                L = length(xEst1);
                N = solverParam.N;
                [~, SNRmiss] = ...
                    SNRInpaintingPerformance(...
                    solutionData.xClean(N:L-N),...
                    problemData.x(N:L-N),...
                    xEst2(N:L-N),...
                    problemData.IMiss(N:L-N));
                SNRClip(kf,kClip,nSolver) = SNRmiss(2);
                
                % normalize and save both the reference and the estimates!
                normX = 1.1*max(abs([xEst1(:);xEst2(:);solutionData.xClean(:)]));
                
                L = min([length(xEst2),length(xEst1),length(solutionData.xClean),length(problemData.x)]);
                xEst1 = xEst1(1:L)/normX;
                xEst2 = xEst2(1:L)/normX;
                xClipped = problemData.x(1:L)/normX;
                xClean = solutionData.xClean(1:L)/normX;
                % FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/DeclippingExperiment/declippingExperiment.m
                % BEGIN: ed8c6549bwf9
                audiowrite(sprintf('%s%s%s%g.wav',expParam.destDir,soundFiles(kf).name(1:end-4),'Est1',clippingLevel), xEst1, fs);
                % END: ed8c6549bwf9
                % FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/DeclippingExperiment/declippingExperiment.m
                % BEGIN: ed8c6549bwf9
                audiowrite(sprintf('%s%s%s%g.wav',expParam.destDir,soundFiles(kf).name(1:end-4),'Est2',clippingLevel), xEst2, fs);
                % END: ed8c6549bwf9
                % FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/DeclippingExperiment/declippingExperiment.m
                % BEGIN: ed8c6549bwf9
                audiowrite(sprintf('%s%s%s%g.wav',expParam.destDir,soundFiles(kf).name(1:end-4),'Clipped',clippingLevel), xClipped, fs);
                % END: ed8c6549bwf9
                % FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/DeclippingExperiment/declippingExperiment.m
                % BEGIN: ed8c6549bwf9
                audiowrite(sprintf('%s%s%s%g.wav',expParam.destDir,soundFiles(kf).name(1:end-4),'Ref',clippingLevel), xClean, fs);
                % END: ed8c6549bwf9
                
                fprintf('\n');
                clear a xEst1 xEst2 xClipped xClean IClipped
                save([expParam.destDir 'clippingExp.mat']);
            end
        end
    end
    
    %% Plot results
    averageSNR = squeeze(mean(SNRClip,1));
    disp(averageSNR)
    figure,
    plot(averageSNR)
    legend(arrayfun(@(x)x.name,expParam.solvers,'UniformOutput',false));
    xlabel('Clipping level')
    ylabel('SNR')
    return
    