function snr = SNR(xRef,xEst)
% Signal-to-noise Ratio
%
% Usage: snr = SNR(xRef,xEst)
%
%
% Inputs:
%          - xRef - reference signal
%          - xEst - estimate signal
%
% Outputs:
%          - snr - SNR
%
%
% -------------------
%
% Audio Inpainting toolbox
% Date: June 28, 2011
% By Valentin Emiya, Amir Adler, Maria Jafari
% This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).
% Signal to noise ratio (SNR)

% Add eps to avoid NaN/Inf values
snr = 10*log10((sum(abs(xRef(:)).^2)+eps)/sum((abs(xRef(:)-xEst(:)).^2)+eps));

return
