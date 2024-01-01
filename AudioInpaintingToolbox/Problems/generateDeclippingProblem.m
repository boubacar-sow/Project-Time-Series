function [problemData, solutionData] = generateDeclippingProblem(x,clippingLevel,GR)
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
%   - problemData.x: clipped signal
%   - problemData.IMiss: boolean vector (same size as problemData.x) that indexes clipped
%   samples
%   - problemData.clipSizes: size of the clipped segments (not necessary
%   for solving the problem)
%   - solutionData.xClean: clean signal (input signal after normalization
%
% Note that the input signal is normalized to 0.9999 (-1 is not allowed in
% wav files) to provide problemData.x and solutionData.xClean.

if nargin<3 || isempty(GR)
    GR = false;
end

%% Normalization
xMax = 0.9999;
solutionData.xClean = x/max(abs(x(:)))*xMax;
clippingLevel = clippingLevel*xMax;

%% Clipping (hard threshold)
problemData.x = min(max(solutionData.xClean,-clippingLevel),clippingLevel);
problemData.IMiss = abs(problemData.x)>=clippingLevel; % related indices

%% Size of the clipped segments
problemData.clipSizes = diff(problemData.IMiss);
if problemData.clipSizes(find(problemData.clipSizes,1,'first'))==-1,problemData.clipSizes = [1;problemData.clipSizes]; end
if problemData.clipSizes(find(problemData.clipSizes,1,'last'))==1,problemData.clipSizes = [problemData.clipSizes;-1]; end
problemData.clipSizes = diff(find(problemData.clipSizes));
problemData.clipSizes = problemData.clipSizes(1:2:end);

%% Optional graphical display
if GR
    
    % Plot histogram of the sizes of the clipped segments
    if ~isempty(problemData.clipSizes)
        figure
        hist(problemData.clipSizes,1:max(problemData.clipSizes))
        title('Size of missing segments')
        xlabel('Size'),ylabel('# of segments')
    end
    
    t = (0:length(solutionData.xClean)-1); % time scale in samples
    
    % Plot original and clipped signals
    figure
    plot(t,solutionData.xClean,'',t,problemData.x,'')
    legend('original','clipped')
    
    % Scatter plot between original and clipped signals
    figure
    plot(solutionData.xClean,problemData.x,'.')
    xlabel('Original signal'),ylabel('Clipped signal')
    
    % Spectrograms
    N = 512;
    w = hann(N);
    fs = 1;
    NOverlap = round(.8*N);
    nfft = 2^nextpow2(N)*2*2;
    figure
    subplot(3,3,[1,4])
    spectrogram(solutionData.xClean,w,NOverlap,nfft,fs,'yaxis')
    title('Original')
    xlim(t([1,end]))
    cl = get(gca,'clim');
    set(gca,'clim',cl);
    subplot(3,3,[1,4]+1)
    spectrogram(problemData.x,w,NOverlap,nfft,fs,'yaxis')
    title('Clipped')
    set(gca,'clim',cl);
    subplot(3,3,[1,4]+2)
    spectrogram(solutionData.xClean-problemData.x,w,NOverlap,nfft,fs,'yaxis')
    title('Error (=original-clipped)')
    set(gca,'clim',cl);
    subplot(3,3,7)
    plot(t,solutionData.xClean,'');xlim(t([1,end]))
    subplot(3,3,8)
    plot(t,solutionData.xClean,'',t,problemData.x,'');xlim(t([1,end]))
    subplot(3,3,9)
    plot(t,solutionData.xClean-problemData.x,'');xlim(t([1,end]))
end

return
