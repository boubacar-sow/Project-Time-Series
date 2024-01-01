function [xClipped, IClipped, xClean, clipSizes] = makeClippedSignal(x,clippingLevel,GR)
% Normalize and clip a signal.
%
% Usage:
%   [xClipped, IClipped, xClean, clipSizes] = makeClippedSignal(x,clippingLevel,GR)
%
% Inputs:
%   - x: input signal (may be multichannel)
%   - clippingLevel: clipping level, between 0 and 1
%   - GR (default: false): flag to generate an optional graphical display
%
% Outputs:
%   - xClipped: clipped signal
%   - IClipped: boolean vector (same size as xClipped) that indexes clipped
%   samples
%   - xClean: clean signal
%   - clipSizes: size of the clipped segments
%
% Note that the input signal is normalized to 0.9999 (-1 is not allowed in
% wav files) to provide xClipped and xClean.
%
% -------------------
%
% Audio Inpainting toolbox
% Date: June 28, 2011
% By Valentin Emiya, Amir Adler, Maria Jafari
% This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).

if nargin<3 || isempty(GR)
    GR = false;
end

%% Normalization
xMax = 0.9999;
xClean = x/max(abs(x(:)))*xMax;
clippingLevel = clippingLevel*xMax;

%% DISABLED - Ramp to produce a clipping level that linearly increases
if 0
    xClean = xClean.*(1:length(xClean))'/length(xClean);
end

%% Clipping (hard threshold)
xClipped = min(max(xClean,-clippingLevel),clippingLevel);
IClipped = abs(xClipped)>=clippingLevel; % related indices

%% Size of the clipped segments
if nargout>3 || GR
   %     clipSizes = diff(find(diff(~IClipped)));
   %     clipSizes = clipSizes(2-(IClipped(1)==0):2:end);
   clipSizes = diff(IClipped);
   if clipSizes(find(clipSizes,1,'first'))==-1,clipSizes = [1;clipSizes]; end
   if clipSizes(find(clipSizes,1,'last'))==1,clipSizes = [clipSizes;-1]; end
   clipSizes = diff(find(clipSizes));
   clipSizes = clipSizes(1:2:end);
end

%% Optional graphical display
if GR
    
    % Plot histogram of the sizes of the clipped segments
    if ~isempty(clipSizes)
        figure
        hist(clipSizes,1:max(clipSizes))
        title('Size of missing segments')
        xlabel('Size'),ylabel('# of segments')
    end
    
    t = (0:length(xClean)-1); % time scale in samples
    
    % Plot original and clipped signals
    figure
    plot(t,xClean,'',t,xClipped,'')
    legend('original','clipped')
    
    % Scatter plot between original and clipped signals
    figure
    plot(xClean,xClipped,'.')
    xlabel('Original signal'),ylabel('Clipped signal')
    
    % Spectrograms
    N = 512;
    w = hann(N);
    fs = 1;
    NOverlap = round(.8*N);
    nfft = 2^nextpow2(N)*2*2;
    figure
    subplot(3,3,[1,4])
    spectrogram(xClean,w,NOverlap,nfft,fs,'yaxis')
    title('Original')
    xlim(t([1,end]))
    cl = get(gca,'clim');
    set(gca,'clim',cl);
    subplot(3,3,[1,4]+1)
    spectrogram(xClipped,w,NOverlap,nfft,fs,'yaxis')
    title('Clipped')
    set(gca,'clim',cl);
    subplot(3,3,[1,4]+2)
    spectrogram(xClean-xClipped,w,NOverlap,nfft,fs,'yaxis')
    title('Error (=original-clipped)')
    set(gca,'clim',cl);
    subplot(3,3,7)
    plot(t,xClean,'');xlim(t([1,end]))
    subplot(3,3,8)
    plot(t,xClean,'',t,xClipped,'');xlim(t([1,end]))
    subplot(3,3,9)
    plot(t,xClean-xClipped,'');xlim(t([1,end]))
end

return
