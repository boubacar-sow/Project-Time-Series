function [SNRAll,SNRmiss] = SNRInpaintingPerformance(xRef,xObs,xEst,IMiss,DISP_FLAG)
% Various SNR measures for inpainting performance
%
% Usage: [SNRAll,SNRmiss] = SNRInpaintingPerformance(xRef,xObs,xEst,IMiss,DISP_FLAG)
%
%
% Inputs:
%          - xRef - reference signal
%          - xObs - observed signal
%          - xEst - estimate signal
%          - IMiss - location of missing data
%
% Outputs:
%          - SNRAll - SNRAll(1) is the original SNR, between xRef and xObs; 
%          SNRAll(2) is the SNR is the obtained SNR, between xRef and xEst
%          - SNRmiss - the same as SNRAll but computed on the missing/restored
%          samples only
%
%
% -------------------
%
% Audio Inpainting toolbox
% Date: June 28, 2011
% By Valentin Emiya, Amir Adler, Maria Jafari
% This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).
if nargin<5
   DISP_FLAG = 0;
end

SNRAll = [SNR(xRef,xObs),SNR(xRef,xEst)];
SNRmiss = [SNR(xRef(IMiss),xObs(IMiss)),SNR(xRef(IMiss),xEst(IMiss))];

if DISP_FLAG>0
   fprintf('SNR on all samples / clipped samples:\n');
   fprintf('Original: %g dB / %g dB\n',...
      SNRAll(1),...
      SNRmiss(1));
   fprintf('Estimate: %g dB / %g dB\n',...
      SNRAll(2),...
      SNRmiss(2));
end

return
