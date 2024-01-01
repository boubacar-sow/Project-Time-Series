function y = inpaintFrame_consOMP_Gabor(problemData,param)
% Inpainting method based on OMP with a constraint
% on the amplitude of the reconstructed samples an optional constraint
% on the maximum value of the clipped samples, and using the Gabor dictionary
% generated by Gabor_Dictionary.m. The method jointly selects
% cosine and sine atoms at the same frequency.
%
% Usage: y = inpaintFrame_consOMP_Gabor(problemData,param)
%
%
% Inputs:
%          - problemData.x: observed signal to be inpainted
%          - problemData.Imiss: Indices of clean samples
%          - param.D - the dictionary matrix (optional if param.D_fun is set)
%          - param.D_fun - a function handle that generates the dictionary 
%          matrix param.D if param.D is not given. See Gabor_Dictionary.m
%          - param.wa - Analysis window
%          - param.Upper_Limit - if present and non-empty this fiels
%          indicates that an upper limit constraint is active and its
%          integer value is such that
%
% Outputs:
%          - y: estimated frame
%
% Note that the CVX library is needed.
%
% -------------------
%
% Audio Inpainting toolbox
% Date: June 28, 2011
% By Valentin Emiya, Amir Adler, Michael Elad, Maria Jafari
% This code is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).




x = problemData.x;
IObs = find(~problemData.IMiss);
p.N = length(x);
E2 = param.OMPerr^2;
E2M=E2*length(IObs);
wa = param.wa(param.N);

% build the dictionary matrix if only the dictionary generation function is given
if ~isfield(param,'D')
    param.D = param.D_fun(param);
end


% clipping level detection
clippingLevelEst = max(abs(x(:)./wa(:)));
IMiss = true(length(x),1);
IMiss(IObs) = false;
IMissPos = find(x>=0 & IMiss);
IMissNeg = find(x<0 & IMiss);

DictPos=param.D(IMissPos,:);
DictNeg=param.D(IMissNeg,:);

% Clipping level: take the analysis window into account
wa_pos = wa(IMissPos);
wa_neg = wa(IMissNeg);
b_ineq_pos = wa_pos(:)*clippingLevelEst;
b_ineq_neg = -wa_neg(:)*clippingLevelEst;
if isfield(param,'Upper_Limit') && ~isempty(param.Upper_Limit)
    b_ineq_pos_upper_limit = wa_pos(:)*param.Upper_Limit*clippingLevelEst;
    b_ineq_neg_upper_limit = -wa_neg(:)*param.Upper_Limit*clippingLevelEst;
else
    b_ineq_pos_upper_limit = Inf;
    b_ineq_neg_upper_limit = -Inf;
end

%%
Dict=param.D(IObs,:);
W=1./sqrt(diag(Dict'*Dict));
Dict=Dict*diag(W);
Dict1 = Dict(:,1:end/2);
Dict2 = Dict(:,end/2+1:end);
Dict1Dict2 = sum(Dict1.*Dict2);
n12 = 1./(1-Dict1Dict2.^2);
xObs=x(IObs);
%K = size(param.D,2);

residual=xObs;
maxNumCoef = param.sparsityDegree;
indx = [];
% currResNorm2 = sum(residual.^2);
currResNorm2 = E2M*2; % set a value above the threshold in order to have/force at least one loop executed
j = 0;
while currResNorm2>E2M && j < maxNumCoef,
    j = j+1;
    proj=residual'*Dict;
    proj1 = proj(1:end/2);
    proj2 = proj(end/2+1:end);
    
    alpha_j = (proj1-Dict1Dict2.*proj2).*n12;
    beta_j = (proj2-Dict1Dict2.*proj1).*n12;
    
    err_j = sum(abs(repmat(residual,1,size(Dict1,2))-Dict1*sparse(diag(alpha_j))-Dict2*sparse(diag(beta_j))).^2);
    [dum pos] = min(err_j);
    
    indx(end+1)=pos;
    indx(end+1)=pos+size(Dict1,2);
    a=pinv(Dict(:,indx(1:2*j)))*xObs;
    residual=xObs-Dict(:,indx(1:2*j))*a;
    currResNorm2=sum(residual.^2);
end;

%% Constrained reestimation of the non-zero coefficients
j = length(indx);
if isinf(b_ineq_pos_upper_limit)
    %% CVX code
    cvx_begin
    cvx_quiet(true)
    variable a(j)
    minimize(norm(Dict(:,indx)*a-xObs))
    subject to
    DictPos(:,indx)*(W(indx).*a) >= b_ineq_pos
    DictNeg(:,indx)*(W(indx).*a) <= b_ineq_neg
    cvx_end
    if cvx_optval>1e3
        cvx_begin
        cvx_quiet(true)
        variable a(j)
        minimize(norm(Dict(:,indx)*a-xObs))
        cvx_end
    end
else
    %% CVX code
    cvx_begin
    cvx_quiet(true)
    variable a(j)
    minimize(norm(Dict(:,indx)*a-xObs))
    subject to
    DictPos(:,indx)*(W(indx).*a) >= b_ineq_pos
    DictNeg(:,indx)*(W(indx).*a) <= b_ineq_neg
    DictPos(:,indx)*(W(indx).*a) <= b_ineq_pos_upper_limit
    DictNeg(:,indx)*(W(indx).*a) >= b_ineq_neg_upper_limit
    cvx_end
    if cvx_optval>1e3
        cvx_begin
        cvx_quiet(true)
        variable a(j)
        minimize(norm(Dict(:,indx)*a-xObs))
        cvx_end
    end
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

