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

% choiceTrials

% params
alpha  = params(1);
beta   = params(2);
beta_c = params(3);

% if (length(params) > 3)
%     decay = params(4);
% else
%     decay = 0;
% end

% Set a fixed trial number to reset Q-values.
resetTrial = flags.resetQ;
if (isnumeric(flags.resetQ))
    resetTrial = flags.resetQ;
end

Q        = zeros(maxTrials, numBandits);
pc       = zeros(1, maxTrials);
rpe      = zeros(1, maxTrials);
runQ     = zeros(numBandits, 1);

for trialIdx = 1:maxTrials

    % Flag: Reset Q values (to median) at context boundaries.
    % if resetTrial is set to (logical) 'true', then this will be trial '1' of the new context.
    % if resetTrial is numeric, then it will be at that trial number.
%     if (resetTrial && mod(trialIdx, 30) == resetTrial)
%         runQ = zeros(numBandits, 1)+0.33;
%     end

%     if (~resetTrial && mod(trialIdx, 30) == 1)
%         % Mix Q-values back towards the mean.
%         runQ = (1-decay)*runQ + decay*0.5;
%     end

    % Save record of Q-values used to make this choice.
    Q(trialIdx, :) = runQ;

    % Bandits are coded 0:2 - set to 1:3 so we can index the Q array.
    chosenBandit = trialrec{trialIdx}.choice+1;
%     chosenBandit
    if (chosenBandit == 0) % Invalid trial. Skip.
        continue;
    end

    if trialIdx > 1
        prevChosenBandit = trialrec{trialIdx-1}.choice + 1;
    else
        prevChosenBandit = -1;
    end

    % Compute choice probability.
    % Was this choice the same as the last one? Perseveration term (1 if yes, 0 if no.)
%     if (trialIdx > 1)
%         lastChoiceTrial = trialIdx-1;
%         if (trialrec{lastChoiceTrial}.type ~= 0)
%             % Go to most recent choice trial, skip over the probe.
%             lastChoiceTrial = trialIdx-2;
%             % XXX: Maybe not?
%         end
% %        persev = ((trialrec{lastChoiceTrial}.choice == [0:2]).*2)-1;   % -1 if no.
%         persev = [trialrec{lastChoiceTrial}.choice == [0:2]];
%         persev = beta_c * persev;
%     else
%         persev = zeros(3,1);
%     end

    nonChosenBandits = find((1:numBandits) ~= chosenBandit);
    otherBandit1 = nonChosenBandits(1);
    otherBandit2 = nonChosenBandits(2);
    I1 = otherBandit1 == prevChosenBandit;
    I2 = otherBandit2 == prevChosenBandit;
    Ic = chosenBandit == prevChosenBandit;

    denom         = exp(beta_c.*I1 + beta*runQ(otherBandit1)) + exp(beta_c.*I2 + beta*runQ(otherBandit2)) + exp(beta_c.*Ic + beta*runQ(chosenBandit));
    pc(trialIdx)  = exp(beta_c.*Ic + beta*runQ(chosenBandit))/denom;

    % Update Q value with outcome.
    rpe(trialIdx) = trialrec{trialIdx}.rwdval - runQ(chosenBandit);
    runQ(chosenBandit)  = runQ(chosenBandit) + alpha * rpe(trialIdx);
end

% pc
% choiceTrials
% pc(choiceTrials)
nloglik = -sum(log(pc(choiceTrials)));

% add in the log prior probability of the parameters
nloglik = nloglik - log(flags.pp_alpha(alpha));
nloglik = nloglik - log(flags.pp_beta(beta));
nloglik = nloglik - log(flags.pp_betaC(beta_c));
end
