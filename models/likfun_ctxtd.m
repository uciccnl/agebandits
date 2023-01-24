function [nloglik Q rpe pc]= likfun_ctxtd(params, trialrec, flags)
%
% Likelihood function for q learning model
%
% INPUTS
%   trialrec
%   params      (alpha - learning rate; beta - softmax temp; beta_p - perseveration)
%               (decay - amount to decay towards prior at context shift)
%

numBandits = 3;
maxTrials  = 180;

% Select all valid choice trials.
choiceTrials    = cellfun(@(x)(x.choice>-1 & x.type==0), trialrec(1:maxTrials));
choiceTrials    = find(choiceTrials);
% choiceTrials    = choiceTrials(1:maxTrials)

% params

alpha  = params(1);
beta   = params(2);
beta_c = params(3);

Q        = zeros(maxTrials, numBandits);
pc       = zeros(1, maxTrials);
rpe       = zeros(1, maxTrials);

pc(1)   = 0.5;

if (length(params) > 3)
    decay = params(4);
else
    decay = 0;
end

% Set a fixed trial number to reset Q-values.
resetTrial = flags.resetQ;
if (isnumeric(flags.resetQ))
    resetTrial = flags.resetQ;
end

Q        = zeros(maxTrials, numBandits);
pc       = zeros(1, maxTrials);
rpe       = zeros(1, maxTrials);
runQ = zeros(numBandits, 1)+0.5;

pc(1)   = 0.5;

if (length(params) > 3)
    decay = params(4);
else
    decay = 0;
end

for trialIdx = 1:maxTrials
    if (trialrec{trialIdx}.type > 0)
        continue;
    end

    % Flag: Reset Q values (to median) at context boundaries.
    % if resetTrial is set to (logical) 'true', then this will be trial '1' of the new context.
    % if resetTrial is numeric, then it will be at that trial number.
    if (resetTrial && mod(trialIdx, 30) == resetTrial)
        runQ = zeros(numBandits, 1)+0.5;
    end

    if (~resetTrial && mod(trialIdx, 30) == 1)
        % Mix Q-values back towards the mean.
        runQ = (1-decay)*runQ + decay*0.5;
    end

    % Save record of Q-values used to make this choice.
    Q(trialIdx, :) = runQ;

    % Bandits are coded 0:2 - set to 1:3 so we can index the Q array.
    thisBandit = trialrec{trialIdx}.choice+1;
    if (thisBandit == 0)
        % Skipped trial.
%        disp(['Skipping ' num2str(trialIdx)]);
        continue;
    end

    % Compute choice probability.
    % Was this choice the same as the last one? Perseveration term (1 if yes, 0 if no.)
    if (trialIdx > 1)
        lastChoiceTrial = trialIdx-1;
        if (trialrec{lastChoiceTrial}.type ~= 0)
            % Go to most recent choice trial, skip over the probe.
            lastChoiceTrial = trialIdx-2;
            % XXX: Maybe not?
        end
%        persev = ((trialrec{lastChoiceTrial}.choice == [0:2]).*2)-1;   % -1 if no.
        persev = [trialrec{lastChoiceTrial}.choice == [0:2]];
        persev = beta_c * persev;
    else
        persev = zeros(3,1);
    end

    denom         = exp(persev(1) + beta*runQ(1)) + exp(persev(2) + beta*runQ(2)) + exp(persev(3) + beta*runQ(3));
    pc(trialIdx)  = exp(persev(thisBandit) + beta*runQ(thisBandit))/denom;

    % Update Q value with outcome.
    runQ(thisBandit)  = runQ(thisBandit) + alpha*(trialrec{trialIdx}.rwdval - runQ(thisBandit));
end

% pc
% choiceTrials
% pc(choiceTrials)
nloglik = -sum(log(pc(choiceTrials)));
% nloglik
% alpha
% beta
% beta_c

% add in the log prior probability of the parameters
nloglik = nloglik - log(flags.pp_alpha(alpha));
nloglik = nloglik - log(flags.pp_beta(beta));
nloglik = nloglik - log(flags.pp_betaC(beta_c));

% nloglik