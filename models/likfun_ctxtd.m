function [pc Qrec] = genreg_ctxQ(trialrec, params, flags)
% function [pc Qrec] genreg_ctxQ(trialrec, params, flags)
%
% Generate Q-values for each context.
%
% INPUTS
%   trialrec
%   params      (alpha - learning rate; beta - softmax temp; beta_p - perseveration)
%               (decay - amount to decay towards prior at context shift)
%
% FLAGS
%   resetQ      (false; Reset Q-values at context-room transitions)
%   reversals   (false; Detect reversals & reset Q-values at those points; XXX unimp)
%

numBandits = 3;
numTrials  = 180;

paramNames = {'alpha', 'beta', 'sticky', 'decay'};
params = transformParams(params, paramNames, 0);

alpha  = params(1);
beta   = params(2);
beta_p = params(3);
if (length(params) > 3)
    decay = params(4);
else
    decay = 0;
end

if nargin < 3
    flags.resetQ = false;
end

% Set a fixed trial number to reset Q-values.
resetTrial = flags.resetQ;
if (isnumeric(flags.resetQ))
    resetTrial = flags.resetQ;
end

pc   = NaN(numTrials, 1);

% Trial-by-trial record of Q values.
Qrec = NaN(numTrials, numBandits);

% Running Q-values - initialize at median.
runQ = zeros(numBandits, 1)+0.5;

for trialIdx = 1:numTrials;
    if (trialrec{trialIdx}.type > 0)
        % Memory probe trial, skip
%        disp(['Skipping ' num2str(trialIdx)]);
        continue;
    end

    % Flag: Reset Q values (to median) at context boundaries.
    % if resetTrial is set to (logical) 'true', then this will be trial '1' of the new context.
    % if resetTrial is numeric, then it will be at that trial number.
    % Last visual reset is the transition to the 7th room (trial 181)
    if (resetTrial && mod(trialIdx, 30) == resetTrial && trialIdx < 182)
        runQ = zeros(numBandits, 1)+0.5;
    end

    if (~resetTrial && mod(trialIdx, 30) == 1 && trialIdx < 182)
        % Mix Q-values back towards the mean.
        runQ = (1-decay)*runQ + decay*0.5;
    end

    % Save record of Q-values used to make this choice.
    Qrec(trialIdx, :) = runQ;

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
        persev = beta_p * persev;
    else
        persev = zeros(3,1);
    end

    denom         = exp(persev(1) + beta*runQ(1)) + exp(persev(2) + beta*runQ(2)) + exp(persev(3) + beta*runQ(3));
    pc(trialIdx)  = exp(persev(thisBandit) + beta*runQ(thisBandit))/denom;

    % Update Q value with outcome.
    runQ(thisBandit)  = runQ(thisBandit) + alpha*(trialrec{trialIdx}.rwdval - runQ(thisBandit));
end
% sum(isnan(pc))
% numTrials
