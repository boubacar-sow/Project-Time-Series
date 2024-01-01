function MissingSampleTopologyExperiment(expParam)
    % For a total number of missing samples C in a frame, create several
    % configuration of B holes with length A, where A*B=C (i.e. the total 
    % number of missing samples is constant). Test several values of C, several
    % solvers. For each C, test all possible combination of (A,B) such that
    % A*B=C.
    % Note that for each combination (A,B), a number of frames are tested at
    % random and SNR results are then averaged.
    %
    % Usage: MissingSampleTopologyExperiment(expParam)
    %
    %
    % Inputs:
    %          - expParam is an optional structure where the user can define
    %          the experiment parameters.
    %          - expParam.soundDir: path to sound directory. All the .wav files
    %          in this directory will be tested at random.
    %          - expParam.destDir: path to store the results.
    %          - expParam.N: frame length
    %          - expParam.NFramesPerHoleSize: number of frames to use for each
    %          testing configuration (A,B). Results are then averaged.
    %          - expParam.totalMissSamplesList: list of all tested values C for
    %          the total number of missing samples in a frame
    %          - expParam.solvers: list of solvers with their parameters
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
    
    %% Set parameters
    if nargin<1
        expParam = [];
    end
    % Path to audio files
    if ~isfield(expParam,'soundDir'),
        expParam.soundDir = '../../Data/testSpeech8kHz_from16kHz/';
    end
    if ~isfield(expParam,'destDir'),
        expParam.destDir = '../../tmp/missSampTopoExp/';
    end
    if ~exist(expParam.destDir,'dir')
        mkdir(expParam.destDir)
    end
    
    
    % frame length
    if ~isfield(expParam,'N')
        expParam.N = 512;
        expParam.N = 256;
        warning('AITB:N','Frame length=256 is used to have faster computations. Recommended frame length is 512 at 8kHz.');
    end
    
    % Number of random frames to test
    if ~isfield(expParam,'NFramesPerHoleSize')
        expParam.NFramesPerHoleSize = 20;
        warning('AITB:NFrames','expParam.NFramesPerHoleSize = 20 is used to have faster computations. Recommended value: several hundreds.');
    end
    
    % Number of missing samples: which numbers to test?
    if ~isfield(expParam,'totalMissSamplesList')
        expParam.totalMissSamplesList = [12,36,60,120,180,240];
        expParam.totalMissSamplesList = [12,36];
        warning('AITB:Miss','expParam.totalMissSamplesList = [12,36] is used to have faster computations. Recommended list: expParam.totalMissSamplesList = [12,36,60,120,180,240].');
    end
    
    % Choose the solver methods you would like to test: OMP, L1, Janssen
    if ~isfield(expParam,'solvers')
        nSolver = 0;
        nSolver = nSolver+1;
        expParam.solvers(nSolver).name = 'OMP-G';
        expParam.solvers(nSolver).inpaintFrame = @inpaintFrame_OMP_Gabor; % solver function
        expParam.solvers(nSolver).param.N = expParam.N; % frame length
        expParam.solvers(nSolver).param.OMPerr = 0.001;
        expParam.solvers(nSolver).param.sparsityDegree = expParam.solvers(nSolver).param.N/4;
        expParam.solvers(nSolver).param.D_fun = @Gabor_Dictionary; % Dictionary (function handle)
        expParam.solvers(nSolver).param.redundancyFactor = 2; % Dictionary redundancy
        expParam.solvers(nSolver).param.wa = @wRect; % Analysis window
        
        nSolver = nSolver+1;
        expParam.solvers(nSolver).name = 'Janssen';
        expParam.solvers(nSolver).inpaintFrame = @inpaintFrame_janssenInterpolation; % solver function
        expParam.solvers(nSolver).param.N = expParam.N; % frame length
    end
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    soundDir = expParam.soundDir;
    wavFiles = dir([soundDir '*.wav']);
    wavFiles = arrayfun(@(x)[soundDir x.name],wavFiles,'UniformOutput',false);
    
    %% Draw a list of random frames
    % Choose an audio file at random
    frameParam.kFrameFile = randi(length(wavFiles),expParam.NFramesPerHoleSize);
    [~, fs] = audioread([soundDir wavFiles{1}]);
    Ne = round(512/16000*fs);
    E2m = zeros(length(wavFiles),1);
    for kf = 1:length(wavFiles)
        [x, ~] = audioread(wavFiles{kf});
        xm = filter(ones(Ne,1)/Ne,1,abs(x.^2));
        E2m(kf) = 10*log10(max(xm));
    end
    
    % Choose the location of a frame at random, with a minimum energy
    maxDiffE2m = 10;
    frameParam.kFrameBegin = NaN(expParam.NFramesPerHoleSize,1);
    for kf = 1:expParam.NFramesPerHoleSize
        % FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/MissingSampleTopologyExperiment/MissingSampleTopologyExperiment.m
        % BEGIN: ed8c6549bwf9
        info = audioinfo(wavFiles{frameParam.kFrameFile(kf)});
        siz = info.TotalSamples;

        % END: ed8c6549bwf9
        while true
            frameParam.kFrameBegin(kf) = randi(siz(1)-expParam.N+1);
            % FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/MissingSampleTopologyExperiment/MissingSampleTopologyExperiment.m
            % BEGIN: ed8c6549bwf9
            x = audioread(wavFiles{frameParam.kFrameFile(kf)},[0,expParam.N-1]+frameParam.kFrameBegin(kf));
            % END: ed8c6549bwf9
            E2m0 = 10*log10(mean(abs(x.^2)));
            if E2m(frameParam.kFrameFile(kf))-E2m0 <= maxDiffE2m
                break
            end
        end
    end
    
    %% Test each number of missing samples
    PerfRes = cell(length(expParam.totalMissSamplesList),length(expParam.solvers));
    factorsToTest = cell(length(expParam.totalMissSamplesList),length(expParam.solvers));
    outputFile = [expParam.destDir 'missSampTopoExp.mat'];
    for kSolver = 1:length(expParam.solvers)
        fprintf('\n ------ Solver: %s ------\n\n',...
            expParam.solvers(kSolver).name);
        for kMiss = 1:length(expParam.totalMissSamplesList)
            NMissSamples = expParam.totalMissSamplesList(kMiss);
            factorsToTest{kMiss} = allFactors(NMissSamples);
            PerfRes{kMiss,kSolver} = zeros([length(factorsToTest{kMiss}),expParam.NFramesPerHoleSize]);
            for kFactor = 1:length(factorsToTest{kMiss})
                holeSize = factorsToTest{kMiss}(kFactor);
                NHoles = NMissSamples/holeSize;
                fprintf('%d %d-length holes (%d missing samples = %.1f%%)\n',...
                    NHoles,holeSize,NMissSamples,NMissSamples/expParam.N*100)
                problemParameters.holeSize = holeSize;
                problemParameters.NHoles = NHoles;
                for kFrame = 1:expParam.NFramesPerHoleSize
                    %% load audio frame
                    % FILEPATH: /home/tp-home007/bsow/Documents/MATLAB/AudioInpaintingToolbox/Experiments/MissingSampleTopologyExperiment/MissingSampleTopologyExperiment.m
                    % BEGIN: ed8c6549bwf9
                    xFrame = audioread(wavFiles{frameParam.kFrameFile(kFrame)},...
                                        frameParam.kFrameBegin(kFrame)+[0,expParam.N-1]);
                    % END: ed8c6549bwf9
                        wavFiles{frameParam.kFrameFile(kFrame)};...
                        frameParam.kFrameBegin(kFrame)+[0,expParam.N-1]; %#ok<VUNUS>
                    
                    %% generate problem
                    [problemData, solutionData] = ...
                        generateMissingGroupsProblem(xFrame,problemParameters);
                    
                    %% solve problem
                    xEst = ...
                        expParam.solvers(kSolver).inpaintFrame(...
                        problemData,...
                        expParam.solvers(kSolver).param);
                    
                    %% compute and store performance
                    [~, SNRmiss] = ...
                        SNRInpaintingPerformance(...
                        solutionData.xClean,...
                        problemData.x,...
                        xEst,...
                        problemData.IMiss);
                    PerfRes{kMiss,kSolver}(kFactor,kFrame) = SNRmiss(2);
                    
                end
            end
            save(outputFile,'PerfRes','expParam');
        end
    end
    
    figure
    Nrows = floor(sqrt(length(expParam.solvers)));
    Ncols = ceil(sqrt(length(expParam.solvers))/Nrows);
    cmap = lines;
    for kSolver = 1:length(expParam.solvers)
        subplot(Nrows,Ncols,kSolver)
        hold on,grid on
        for kMiss = 1:length(expParam.totalMissSamplesList)
            plot(factorsToTest{kMiss},mean(PerfRes{kMiss,kSolver},2),...
                'color',cmap(kMiss,:));
        end
        title(expParam.solvers(kSolver).name)
    end
    saveas(gcf,'myfigure.png')
    
    function m = allFactors(n)
    % Find the list of all factors (not only prime factors)
    
    primeFactors = factor(n);
    
    degrees = zeros(size(primeFactors));
    
    for k=1:length(degrees)
        degrees(k) = sum(primeFactors==primeFactors(k));
    end
    
    [primeFactors, I] = unique(primeFactors);
    degrees = degrees(I);
    
    D = (0:degrees(1)).';
    for k=2:length(degrees)
        Dk = ones(size(D,1),1)*(0:degrees(k));
        D = [repmat(D,degrees(k)+1,1),Dk(:)];
    end
    
    m = unique(sort(prod((ones(size(D,1),1)*primeFactors).^D,2)));
    
    return
    