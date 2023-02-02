function results = fit_model()

mode = '';
likfunToUse = 3; % 1, 2 or 3.
startSubject = 1;
iterations = 10;
saveFile = strcat('resultmatfiles/ctxhybrid_results', mode)

verbose = 1;
veryverbose = 1;
dataDir = strcat('agebandits/data', mode)
dataPattern = 'transformed_Data*.mat';

% set up data
dd = dir(fullfile(dataDir, dataPattern));
nSubs = length(dd);

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
elseif likfunToUse == 3 %hybrid model
    disp('Fitting Hybrid (TD+Sampler) Model');
end

% Alpha - learning /decay rate
param(1).name = 'alpha';
%Beta - explore/exploit tradeoff
param(2).name = 'beta';
%BetaC - perseverative parameter
param(3).name = 'beta_c';

if likfunToUse == 3 %hybrid model
    param(1).name = 'alpha';        % Sampler
    param(2).name = 'beta';

    param(4).name = 'alpha';        % TD
    param(5).name = 'beta';
end

if likfunToUse == 0
    paramNames = {'beta_c'};
elseif likfunToUse == 1
    paramNames = {'alpha', 'beta', 'beta_c'};
elseif likfunToUse == 2
    paramNames = {'alpha', 'beta', 'beta_c'};
elseif likfunToUse == 3
    paramNames = {'alpha', 'beta', 'beta_c', 'alpha', 'beta'};
end


% % Generate transformed ranges for random starts below
% for paramIdx = 1:length(param);
%     transformerIndex = find(cellfun(@(x)(streq(x, param(paramIdx).name)), {paramTable{:,1}}));
%     paramRange = paramTable{transformerIndex,2}{:};
%     paramEdge = paramTable{transformerIndex,3}{:};
%     % Want to get the endpoint just off the edge, so it doesn't crash.
%     param(paramIdx).lb = transform_params(paramRange(1)+paramEdge, {param(paramIdx).name}, 1);
%     param(paramIdx).ub = transform_params(paramRange(2)-paramEdge, {param(paramIdx).name}, 1);
%     param(paramIdx)
% end

for paramIdx = 1:length(param)
    idx = find(cellfun(@(x)(streq(x, param(paramIdx).name)), {paramTable{:,1}}));
    paramRange = paramTable{idx,2}{:};
    param(paramIdx).lb = paramRange(1);
    param(paramIdx).ub = paramRange(2);
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

for sub = startSubject:nSubs
    ds1 = datetime;

    disp([newline '>>> Fitting subject ' int2str(sub) ' ' datestr(datetime)]);

    [dataToFit{sub}, filename] = loadSubj(sub, dataDir, dd);

    % Set up results structure
    results{sub}.numParams = numParams;
    results{sub}.nLogLik = Inf;
    results{sub}.file = filename;

    nUnchanged = 0;
    starts = 0;

    if likfunToUse == 1
        flags.resetQ = false;
        flags.signed_LR = false;
        f = @(x) likfun_ctxtd(transform_params(x, paramNames), dataToFit{sub}.trialrec, flags);
    elseif likfunToUse == 2
        flags.numSamples = 1;
        precomputed = load(strcat('precomputed/precomputed_sub', num2str(sub), '_', num2str(flags.numSamples), '.mat'));
        flags.choicerec = precomputed.choicerec;
        flags.combs = precomputed.combs;
        f = @(x) likfun_ctxsampler(transform_params(x, paramNames), dataToFit{sub}.trialrec, flags);
    elseif likfunToUse == 3
        flags.resetQ = false;
        flags.numSamples = 1;
        precomputed = load(strcat('precomputed/precomputed_sub', num2str(sub), '_', num2str(flags.numSamples), '.mat'));
        flags.choicerec = precomputed.choicerec;
        flags.combs = precomputed.combs;
        f = @(x) likfun_ctxhybrid(transform_params(x, paramNames), dataToFit{sub}.trialrec, flags);
    end

    while nUnchanged < iterations       % "Convergence" test. This could be better.
        d1 = datetime;
        starts = starts + 1;    % add 1 to starts
%         starts

        %set fminunc starting values
        x0 = zeros(1,numParams); % initialize at zeros
        for p = 1:numParams
            x0(p) = unifrnd(param(p).lb, param(p).ub); %pick random starting values
        end

        % find min negative log likelihood = maximum likelihood for each subject
        transformed_x0 = transform_params(x0, paramNames, 1);    % to raw space for input to fminunc
        [xf,nloglik,exitflag,output,~,~] = fminunc(f, transformed_x0, options);
        transformed_xf = transform_params(xf, paramNames);      % to valid space output of fminunc

        if (verbose == 1)
            disp(['> valid_x0=[' num2str(x0) ']  raw_x0=[' num2str(transformed_x0) ']  raw_xf=[' num2str(xf) '] valid_xf=[' num2str(transformed_xf) ']']);
        end
        
        d2 = datetime;
        time_taken = round(minutes(d2-d1), 2);

%         if exitflag ~= 1
%             disp("oop")
%             [xf, nloglik,exitflag,output] = fminsearch(f, x0, searchopts);
%         end
%         
        if exitflag ~= 1
            disp('Failed to converge')
        else
            if (veryverbose == 1)
                disp(['subject ' num2str(sub) ': start ' num2str(starts) '(' num2str(nUnchanged) '): NLL ' ...
                    num2str(nloglik) ', params: [' num2str(xf) ...
                    '], tr-params: [' num2str(transformed_xf) '] Time: ' num2str(time_taken) 'mins' ]);
            end
        end

        % store min negative log likelihood and associated parameter values
        if (starts == 1 || nloglik < results{sub}.nLogLik)

            if (verbose == 1)
                disp(['NEW best params: subject ' num2str(sub) ', NLL: ' num2str(nloglik) ...
                    ', Previous NLL: ' num2str(results{sub}.nLogLik) ...
                    ', params: [' num2str(xf) '], tr-params: [' num2str(transformed_xf) ']']);
            end

            nUnchanged = 0; %reset to 0 if neg likelihood decreases

            results{sub}.nLogLik  = nloglik;
            results{sub}.params = xf;
            results{sub}.transformedParams = transformed_xf;
            results{sub}.model = likfunToUse;
            results{sub}.exitflag = exitflag;
            results{sub}.output = output;
%             results{sub}.grad(sub,:) = grad;
%             results{sub}.hessian(sub,:,:,:,:) = hessian;
%             results{sub}.laplace(sub) = nloglik + (0.5*length(x)*log(2*pi)) - 0.5*log(det(hessian));
            [~, Q, rpe, pc] = f(xf);        % Run it again to get the Q and pc
            results{sub}.runQ = Q;
            results{sub}.pc = pc;
            results{sub}.rpe = rpe;

            % When computing AIC/BIC we have to take back out the prior probabilities of the parameters.
            useLogLik = nloglik;
            % XXX: Autogenerate these
            if (~isinf(log(flags.pp_alpha(xf(1)))) && ~isnan(log(flags.pp_alpha(xf(1)))))
                useLogLik = useLogLik + log(flags.pp_alpha(xf(1)));
            end
            if (~isinf(log(flags.pp_beta(xf(2)))) && ~isnan(log(flags.pp_beta(xf(2)))))
                useLogLik = useLogLik + log(flags.pp_beta(xf(2)));
            end
            if (~isinf(log(flags.pp_betaC(xf(3)))) && ~isnan(log(flags.pp_betaC(xf(3))))) %betaC
                useLogLik = useLogLik + log(flags.pp_betaC(xf(3)));
            end
            if likfunToUse == 3 % handle additional params for hybrid
                 if (~isinf(log(flags.pp_alpha(xf(4)))) && ~isnan(log(flags.pp_alpha(xf(4))))) %alphaTD
                    useLogLik = useLogLik + log(flags.pp_alpha(xf(4)));
                 end
                 if (~isinf(log(flags.pp_beta(xf(5)))) && ~isnan(log(flags.pp_beta(xf(5))))) %betaTD
                    useLogLik = useLogLik + log(flags.pp_beta(xf(5)));
                 end
            end

            results{sub}.useLogLik = useLogLik;
            results{sub}.AIC = 2*length(xf) + 2*useLogLik;
            results{sub}.BIC = 0.5*length(xf)*log(180) + useLogLik;
        else
            nUnchanged = nUnchanged + 1;
        end  % if nloglik < results{sub}.nloglik
    end % while

%     'Final'
%     strcat('Subject ', num2str(sub))
%     strcat(': Final MAP ', num2str(results.nLogLik(sub)))
%     strcat('final AIC ', num2str(results.AIC(sub)))
%     strcat('final BIC ', num2str(results.BIC(sub)))
%     strcat('final params ', num2str(results.params(sub, :)))
    save(strcat(saveFile, num2str(startSubject), '_', num2str(sub)), 'results');
    ds2 = datetime;
    time_taken = round(minutes(ds2-ds1), 2);
    disp(['>> Subject ' num2str(sub)  ' took a total of ' num2str(time_taken) 'mins' ]);
end

disp([newline '>>>> Done! ' datestr(datetime)]);
