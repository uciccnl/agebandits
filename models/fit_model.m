function results = fit_model(likfunToUse)
%
% This function takes as input a data set, a likelihood function
% and the name where the data should be saved
%
%   likfunToUse = 2 (currently supported values = [2])
%

verbose = 1;
veryverbose = 1;
mode = '_test'
dataDir = strcat('../transformed_Data_06.13.22_FILES', mode)
saveFile = strcat('ctxsample_results', mode)
dataPattern = 'transformed_Data*.mat';

dd = dir(fullfile(dataDir, dataPattern));
submat = [1:length(dd)];

for s = submat;
    dataToFit{s} = loadSubj(s, dataDir, dataPattern);
end

analysis_constants;

% Prior distributions for parameters
% XXX: Auto-generate these from param table
flags.pp_alpha = @(x)(pdf('beta', x, 1.1, 1.1));                  % Beta prior for \alphas (from Daw et al 2011 Neuron)
% flags.pp_pi    = @(x)(pdf('beta', abs(x), 1.1, 1.1));             % symmetric Beta prior for \alpha bump (from Daw et al 2011 Neuron)

% flags.pp_beta = @(x)(pdf('gamma', x, 1.2, 5));                  % Gamma prior for softmax \beta (from Daw et al 2011 Neuron)
low_bound = paramTable{2,2}{1}(1);
upp_bound = paramTable{2,2}{1}(2);
flags.pp_beta  = @(x)(pdf('normal', (x-low_bound)/(upp_bound-low_bound), 0, 10));            % Beta prior for \alphas (from Daw et al 2011 Neuron)

low_bound = paramTable{3,2}{1}(1);
upp_bound = paramTable{3,2}{1}(2);
flags.pp_betaC = @(x)(pdf('normal', (x-low_bound)/(upp_bound-low_bound), 0, 10));            % Beta prior for \alphas (from Daw et al 2011 Neuron)


%% Set up parameter space
% XXX: Auto-generate these from param table

if (likfunToUse == 1)
    disp('Fitting TD Model');
elseif (likfunToUse == 2)
    disp('Fitting Sampler Model');
end

paramNames = {'alpha', 'beta', 'beta_c'};
% Alpha - learning /decay rate
param(1).name = 'alpha';
%Beta - explore/exploit tradeoff
param(2).name = 'beta';
%BetaC - perseverative parameter
param(3).name = 'beta_c';

if likfunToUse == 3 %hybrid model
    disp('Fitting Hybrid (TD+Sampler) Model');
    param(1).name = 'alpha';        % Sampler
    param(2).name = 'beta';

    param(4).name = 'alpha';        % TD
    param(5).name = 'beta';
end

% Generate transformed ranges for random starts below
for paramIdx = 1:length(param);
    transformerIndex = find(cellfun(@(x)(streq(x, param(paramIdx).name)), {paramTable{:,1}}));
    paramRange = paramTable{transformerIndex,2}{:};

    % Want to get the endpoint just off the edge, so it doesn't crash.
    param(paramIdx).lb = transform_params(paramRange(1)+1e-5, {param(paramIdx).name}, 1);
    param(paramIdx).ub = transform_params(paramRange(2)-1e-5, {param(paramIdx).name}, 1);
    param(paramIdx)
end

%% Important things to pass to fminunc
numParams = length(param); %specify number of parameters
% lb = [param.lb]; %specify lower and upper bounds of parameters
% ub = [param.ub];

% define options for fminunc
options = optimset('Display','off');
searchopts  = optimset('Display','off','TolCon',1e-6,'TolFun',1e-5,'TolX',1e-5,...
                       'DiffMinChange',1e-4,'Maxiter',1000,'MaxFunEvals',2000);

% Set up results structure
results.numParams = numParams;

% for sub = 1:nSubs
for sub = submat;
    disp(['Fitting subject ' int2str(sub)]);

    nUnchanged = 0;
    starts = 0;

    if likfunToUse == 1
        flags.resetQ = false;
        f = @(x) likfun_ctxtd(transform_params(x, paramNames), dataToFit{sub}.trialrec, flags);
    elseif likfunToUse == 2
        flags.numSamples = 1;
        precomputed = load(strcat('precomputed/precomputed_sub', num2str(sub), '_', num2str(flags.numSamples), '.mat'));
        flags.choicerec = precomputed.choicerec;
        flags.combs = precomputed.combs;
        f = @(x) likfun_ctxsampler(transform_params(x, paramNames), dataToFit{sub}.trialrec, flags);
    end

    while nUnchanged < 50       % "Convergence" test. This could be better.
        starts = starts + 1;    % add 1 to starts
%         starts

        %set fminunc starting values
        x0 = zeros(1,numParams); % initialize at zers
        for p = 1:numParams
            x0(p) = unifrnd(param(p).lb, param(p).ub); %pick random starting values
        end
        % find min negative log likelihood = maximum likelihood for each
        % subject
%         x0
%         f(x0)
        [x,nloglik,exitflag,output,~,~] = fminunc(f, x0, options);
%         if exitflag ~= 1
%             disp("oop")
%             [x, nloglik,exitflag,output] = fminsearch(f, x0, searchopts);
%         end
%         
        if exitflag ~= 1
%             disp(['Failure to converge']);
            continue;
        end

        if (veryverbose == 1)
            disp(['subject ' num2str(sub) ': start ' num2str(starts) '(' num2str(nUnchanged) '): NLL ' num2str(nloglik) ', params [' num2str(x) '], tr-params [' num2str(transform_params(x, paramNames)) ']']);
        end
        % store min negative log likelihood and associated parameter values
        if ((isfield(results, 'nLogLik') == false) || (nloglik < results.nLogLik(sub) - 0.01))

            if (verbose == 1)
                disp(['NEW: subject ' num2str(sub) ', params ' num2str(x) ', new best params: ' ...
                    num2str(transform_params(x, paramNames)) ', NLL: ' num2str(nloglik)]);
            end
        
            nUnchanged = 0; %reset to 0 if likelihood changes

            results.nLogLik(sub)  = nloglik;
            results.params(sub, :) = x;
            results.transformedParams(sub, :) = transform_params(x, paramNames);

            useLogLik = nloglik;

            results.useLogLik(sub) = useLogLik;
            results.AIC(sub) = 2*length(x) + 2*useLogLik;
            results.BIC(sub) = 0.5*length(x)*log(180) + useLogLik;
            results.model(sub) = likfunToUse;
            results.exitflag(sub) = exitflag;
            results.output(sub) = output;
%             results.grad(sub,:) = grad;
%             results.hessian(sub,:,:,:,:) = hessian;
%             results.laplace(sub) = nloglik + (0.5*length(x)*log(2*pi)) - 0.5*log(det(hessian));
            [~, Q, rpe, pc] = f(x);        % Run it again to get the Q and pc
            results.runQ{sub} = Q;
            results.pc{sub} = pc;
            results.rpe{sub} = rpe;
        else
            nUnchanged = nUnchanged + 1;
        end  % if starts == 1
    end % while
% 
%     'Final'
%     strcat('Subject ', num2str(sub))
%     strcat(': Final MAP ', num2str(results.nLogLik(sub)))
%     strcat('final AIC ', num2str(results.AIC(sub)))
%     strcat('final BIC ', num2str(results.BIC(sub)))
%     strcat('final params ', num2str(results.params(sub, :)))
    save(saveFile, 'results');

end
