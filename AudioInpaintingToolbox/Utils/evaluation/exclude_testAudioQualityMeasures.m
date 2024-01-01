function testAudioQualityMeasures
%
%
% Usage:
%
%
% Inputs:
%          - 
%          - 
%          - 
%          - 
%          - 
%          - 
%          - 
%          - 
%
% Outputs:
%          - 
%          - 
%          - 
%          - 
%
% Note that the CVX library is needed.
%
% -------------------
%
% Audio Inpainting toolbox
% Date: June 28, 2011
% By Valentin Emiya, Amir Adler, Maria Jafari
% This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).

soundDir = './';

[xRef fs] = wavread([soundDir 'xClean.wav']);

testFiles = {'xClipped.wav','xEst2.wav','xEstInterp'};
Nf = length(testFiles);

SNR = zeros(Nf,1);
PSM = zeros(Nf,1);
PSMt = zeros(Nf,1);
PESQ_MOS = zeros(Nf,1);
EAQUAL_ODG = zeros(Nf,1);
EAQUAL_DIX = zeros(Nf,1);

options.ENABLE_PEMOQ = true;

for kf = 1:Nf
    xTest = wavread([soundDir testFiles{kf}]);
    [SNR(kf) PSM(kf),PSMt(kf),...
        PESQ_MOS(kf),EAQUAL_ODG(kf), EAQUAL_DIX(kf)] = ...
        audioQualityMeasures(xRef,xTest,fs,options);
end

for kf = 1:Nf
    fprintf('Quality of %s: SNR = %g dB, PSM=%g, PSMt=%g, PESQ=%g, EAQUAL_ODG=%g, EAQUAL_DIX=%g\n',...
        testFiles{kf},SNR(kf),PSM(kf),PSMt(kf),PESQ_MOS(kf),EAQUAL_ODG(kf),EAQUAL_DIX(kf));
end

Q = [SNR,PSM,PSMt,PESQ_MOS,EAQUAL_ODG,EAQUAL_DIX];
Qs = {'SNR','PSM','PSMt','PESQ MOS','EAQUAL ODG','EAQUAL DIX'};

figure
for k=1:size(Q,2)
    subplot(ceil(sqrt(size(Q,2))),ceil(sqrt(size(Q,2))),k)
    plot(Q(:,k))
    xlabel('audio files');
    ylabel(Qs{k})
end
return
