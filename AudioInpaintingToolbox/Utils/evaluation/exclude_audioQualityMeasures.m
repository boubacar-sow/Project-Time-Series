function [SNRx, PSM, PSMt, PESQ_MOS, EAQUAL_ODG, EAQUAL_DIX] = ...
   audioQualityMeasures(xRef,xTest,fs,options)
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
% Computes a series of audio quality measures
%
% Usage:
%    [PSM,PSMt] = ...
%         audioQualityMeasures(xRef,xTest,fs,options)
%
% Inputs:
%    - xRef: reference signal
%    - xTest: test signal (to be compared to xRef)
%    - fs: sampling frequency
%    - optional parameters [default]:
%        - options.ENABLE_PEMOQ: flag to use PEMO-Q or not [true]
%        - options.pemoQExec: name of PEMO-Q executable ['"./PEMO-Q v1.1.2 demo/audioqual_demo.exe"']
%        - options.PESQExec: name of PESQ executable [''./PESQ/pesq'']
%        - options.EAQUALExec: name of EAQUAL (PEAQ) executable ['./EAQUAL/eaqual.exe']
%
% Note that PEMO-Q and EAQUAL programs are Windows executable and that
% under unix, they can be used by means of wine (Windows Emulator). One
% just have to have wine installed.
%
% Valentin Emiya, INRIA, 2010.


%% default options
defaultOptions.pemoQExec = '"C:/Program Files/PEMO-Q v1.2/audioqual.exe"'; % Full licensed version
if isempty(dir(defaultOptions.pemoQExec))
   defaultOptions.pemoQExec = '"./PEMO-Q v1.1.2 demo/audioqual_demo.exe"'; % Demo version
end
defaultOptions.ENABLE_PEMOQ = true;
defaultOptions.PESQExec = './PESQ/pesq';

if ispc
   defaultOptions.EAQUALExec = './EAQUAL/eaqual.exe';
else % if unix, use wine
   defaultOptions.EAQUALExec = 'wine ./EAQUAL/eaqual.exe';
   defaultOptions.pemoQExec = ['wine ' defaultOptions.pemoQExec];
end

if nargin<4
   options = defaultOptions;
else
   names = fieldnames(defaultOptions);
   for k=1:length(names)
      if ~isfield(options,names{k}) || isempty(options.(names{k}))
         options.(names{k}) = defaultOptions.(names{k});
      end
   end
end

if ~ischar(xRef) && ~ischar(xTest) && length(xRef)~=length(xTest)
   warning('EVAL:LENGTH','Different lengths');
   L = min(length(xRef),length(xTest));
   xRef = xRef(1:L);
   xTest = xTest(1:L);
end

if ischar(xRef)
   refFile = xRef;
   sRef = wavread(refFile);
else
   refFile = [tempname '.wav'];
   sRef = xRef;
   wavwrite(xRef,fs,refFile);
end
if ischar(xTest)
   testFile = xTest;
   sTest = wavread(testFile);
else
   testFile = [tempname '.wav'];
   sTest = xTest;
   wavwrite(xTest,fs,testFile);
end


SNRx = SNR(sRef,sTest);

try
   %     if ispc && options.ENABLE_PEMOQ
   if options.ENABLE_PEMOQ
      %% PEMO-Q
      [PSM,PSMt] = aux_pemoq(refFile,testFile,options);
   else
      warning('audioQualityMeasures:noPQ','PEMO-Q is not available (requires Windows plateform)')
      PSM = NaN;
      PSMt = NaN;
   end
   
   %% PESQ
   PESQ_MOS = aux_pesq(refFile,testFile,options);
   
   %% EAQUAL (PEAQ)
   [EAQUAL_ODG, EAQUAL_DIX] = aux_eaqual(refFile,testFile,options);
   
   if ~ischar(xRef)
      %% Delete temporary files
      delete(refFile);
   end
   if ~ischar(xTest)
      delete(testFile);
   end
   
catch
   if ~ischar(xRef)
      %% In case of error, delete the temporary files
      delete(refFile);
   end
   if ~ischar(xTest)
      delete(testFile);
   end
   rethrow;
end


return

function [PSM,PSMt] = aux_pemoq(refFile,testFile,options)
if ~isempty(findstr(options.pemoQExec, 'demo'))
   fprintf('To unlock PEMO-Q demo, please enter the PIN shown in the new window\n');
end
[dum, pemo] = system(sprintf('%s %s %s [] [] 0 0 0', options.pemoQExec, refFile, testFile));
pemo = regexp(pemo, 'PSM.? = \d*.\d*', 'match');
PSM = str2double(cell2mat(regexp(pemo{1},'\d+.?\d*', 'match')));
PSMt = str2double(cell2mat(regexp(pemo{2},'\d+.?\d*', 'match')));

return

function PESQ_MOS = aux_pesq(refFile,testFile,options)
[dum fs] = wavread(refFile,'size');
if ~ismember(fs,[8000 16000])
   error('audioQualityMeasures:badFs',...
      '8kHz or 16 kHz sampling frequency required for PESQ');
end
[dum,s] = system(sprintf('%s +%d %s %s',options.PESQExec,fs,refFile,testFile));
PESQ_MOS = regexp(s, 'Prediction : PESQ_MOS = \d*.\d*', 'match');
PESQ_MOS = str2double(PESQ_MOS{end}(length('Prediction : PESQ_MOS = ')+1:end));
return

function [EAQUAL_ODG, EAQUAL_DIX] = aux_eaqual(refFile,testFile,options)
[dum fs] = wavread(refFile,'size');
DELETE_FLAG = false;
if fs<44100
   warning('EAQUAL:BAD_FS',...
      'Sampling frequency is too low for Eaqual (<44.1kHz).\nResampling first (result not relevant)');
   DELETE_FLAG = true;
   
   x = wavread(refFile);
   fsEaqual = 48000;
   x = resample(x,fsEaqual,fs);
   refFile = [tempname '.wav'];
   wavwrite(x,fsEaqual,refFile);
   
   x = wavread(testFile);
   fsEaqual = 48000;
   x = resample(x,fsEaqual,fs);
   testFile = [tempname '.wav'];
   wavwrite(x,fsEaqual,testFile);
   
   fs = fsEaqual;
end

[dum,s] = system(sprintf('%s -fref %s -ftest %s  -srate %d',options.EAQUALExec,refFile,testFile,fs));

EAQUAL_ODG = regexp(s, 'Resulting ODG:\t.?\d*(\.\d*)?', 'match');
EAQUAL_ODG = str2double(EAQUAL_ODG{end}(length('Resulting ODG: ')+1:end));
EAQUAL_DIX = regexp(s, 'Resulting DIX:\t.?\d*(\.\d*)?', 'match');
EAQUAL_DIX = str2double(EAQUAL_DIX{end}(length('Resulting DIX: ')+1:end));

if DELETE_FLAG
   delete(refFile);
   delete(testFile);
end
return
