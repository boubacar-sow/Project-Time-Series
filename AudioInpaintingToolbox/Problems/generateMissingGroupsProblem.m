function [problemData, solutionData] = generateMissingGroupsProblem(xFrame,problemParameters)
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

% Generate a clipping problem: normalize and clip a signal.
%
% Usage:
%   [problemData, solutionData] = makeClippedSignal(x,clippingLevel,GR)
%
% Inputs:
%   - x: input signal (may be multichannel)
%   - clippingLevel: clipping level, between 0 and 1
%   - GR (default: false): flag to generate an optional graphical display
%
% Outputs:
%   - problemData.xClipped: clipped signal
%   - problemData.IClipped: boolean vector (same size as problemData.xClipped) that indexes clipped
%   samples
%   - problemData.clipSizes: size of the clipped segments (not necessary
%   for solving the problem)
%   - solutionData.xClean: clean signal (input signal after normalization
%
% Note that the input signal is normalized to 0.9999 (-1 is not allowed in
% wav files) to provide problemData.xClipped and solutionData.xClean.

%function [xFrame,xFrameObs,Imiss] = aux_getFrame(xFrame,holeSize,NHoles)%,param,frameParam,kFrame)

N = length(xFrame); % frame length

% Load frame
% xFrame = wavread(param.wavFiles{frameParam.kFrameFile(kFrame)},frameParam.kFrameBegin(kFrame)+[0,N-1]);

% Window
%xFrame = xFrame.*param.wa(N).';

% Normalize
xFrame = xFrame/max(abs(xFrame));

% Build random measurement matrix with NHoles of length holeSize
[M IMiss] = makeRandomMeasurementMatrix(N,problemParameters.NHoles,problemParameters.holeSize);
xFrameObs = xFrame;
xFrameObs(IMiss) = 0;

problemData.x = xFrameObs;
problemData.IMiss = IMiss;
solutionData.xClean = xFrame;
return


function [M Im] = makeRandomMeasurementMatrix(N,NMissingBlocks,blockSize)
% [M Im] = makeRandomMeasurementMatrix(N,NMissingBlocks,blockSize)
%    Create a random measurement matrix M where NMissingBlocks blocks with 
%    size blockSize each are randomly inserted. The boolean vector Im 
%    indicates the location of the NMissingBlocks*blockSize missing
%    samples.
%    If the number of missing samples is large, there may be very few solutions
%    so that after a few failing attempts, the results is generated in a deterministic
%    way (groups separated by one sample). This happens for example when the number of
%    missing samples is close to half the frame length, for isolated samples (blockSize=1).

nTry = 1;
while true
   try
      Im = false(N,1);
      
      possibleStart = 1:N-blockSize+1;
      
      for k=1:NMissingBlocks
         if isempty(possibleStart)
            error('makeRandomMeasurementMatrix:tooMuchMissingSamples',...
               'Too much missing segments');
         end
         I = ceil(rand*(length(possibleStart)-1));
         I = possibleStart(I);
         Im(I+(0:blockSize-1)) = true;
         possibleStart(possibleStart>=I-blockSize & possibleStart<=I+blockSize) = [];
      end
      break
   catch
      fprintf('makeRandomMeasurementMatrix:retry (%d)\n',nTry);
      nTry = nTry+1;
      if nTry>10
            Im = [true(blockSize,NMissingBlocks);false(1,NMissingBlocks)];
            Im = Im(:);
            while length(Im)<N
               N0 = sum(~Im);
               I0 = find(~Im,randi(N0),'first');
               I0 = I0(end);
               Im = [Im(1:I0);false;Im(I0+1:end)];
            end
            Im = circshift(Im,randi(N));
         break;
      end
   end
end
M = eye(N);
M(Im,:) = [];

return
