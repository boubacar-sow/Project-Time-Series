function y = inpaintFrame_OMP(problemData,param)
% Inpainting method based on OMP 
%
% Usage: y = inpaintFrame_OMP(problemData,param)
%
%
% Inputs:
%          - problemData.x: observed signal to be inpainted
%          - problemData.Imiss: Indices of clean samples
%          - param.D - the dictionary matrix (optional if param.D_fun is set)
%          - param.D_fun - a function handle that generates the dictionary 
%          matrix param.D if param.D is not given. See, e.g., DCT_Dictionary.m and Gabor_Dictionary.m
%          - param.wa - Analysis window
%
% Outputs:
%          - y: estimated frame
%
%
% -------------------
%
% Audio Inpainting toolbox
% Date: June 28, 2011
% By Valentin Emiya, Amir Adler, Michael Elad, Maria Jafari
% This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).
% ========================================================
% To do next: use a faster implementation of OMP

%% Load data and parameters

x = problemData.x;
IObs = find(~problemData.IMiss);
p.N = length(x);
   E2 = param.OMPerr^2;
   E2M=E2*length(IObs);

   wa = param.wa(param.N);


%% Build and normalized dictionary
% build the dictionary matrix if only the dictionary generation function is given
if ~isfield(param,'D')
   param.D = param.D_fun(param);
end
Dict=param.D(IObs,:);
W=1./sqrt(diag(Dict'*Dict));
Dict=Dict*diag(W);
xObs=x(IObs);

%% OMP iterations
residual=xObs;
maxNumCoef = param.sparsityDegree;
indx = [];
currResNorm2 = E2M*2; % set a value above the threshold in order to have/force at least one loop executed
j = 0;
while currResNorm2>E2M && j < maxNumCoef,
   j = j+1;
   proj=Dict'*residual;
   [dum pos] = max(abs(proj));
   indx(j)=pos;
   
   a=pinv(Dict(:,indx(1:j)))*xObs;
   
   residual=xObs-Dict(:,indx(1:j))*a;
   currResNorm2=sum(residual.^2);
end

%% Frame Reconstruction
indx(length(a)+1:end) = [];

Coeff = sparse(size(param.D,2),1);
if (~isempty(indx))
   Coeff(indx) = a;
   Coeff = W.*Coeff;
end
y = param.D*Coeff;

return
