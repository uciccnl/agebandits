function [nloglik Q_td rpe_td pc]= likfun_ctxhybrid(params, trialrec, flags)
%
% Likelihood function for memory sampling + TD model

% INPUTS
%   trialrec
%   params      (alpha - learning rate; beta - softmax temp; beta_p - perseveration)
%
%
% Flags:
%   .combs      -> Precomputed k-combinations of each trial index (cell
%   array of numTrials x numSamples containing matrices of size (n+k-1 choose k) x k for each Bandit)
%   .numSamples -> Number of samples to draw (aka: k)
%   .choicerec  -> list of choices made [bandit rwdval isprobe]
%%

% trialrec{1}

numBandits = 3;
maxTrials  = 180;
numSamples = flags.numSamples;

% Select all valid choice trials.
choiceTrials    = cellfun(@(x)(x.choice>-1 & x.type==0), trialrec(1:maxTrials));
choiceTrials    = find(choiceTrials);
% choiceTrials    = choiceTrials(1:maxTrials)

% averageQ = 0;

alpha  = params(1);
beta   = params(2);
beta_c = params(3);
alpha_td = params(4);
beta_td = params(5);

combs     = flags.combs{numSamples};
choicerec = flags.choicerec;

% Q        = zeros(maxTrials, numBandits);
% rpe      = zeros(1, maxTrials);

Q_td     = zeros(maxTrials, numBandits);
rpe_td   = zeros(1, maxTrials);
runQ     = zeros(numBandits, 1);

pc       = zeros(1, maxTrials);

%%
for trialIdx = 1:maxTrials
%     trialIdx

    chosenBandit = trialrec{trialIdx}.choice + 1;
    reward = double(trialrec{trialIdx}.rwdval);

    if trialIdx > 1
        prevChosenBandit = trialrec{trialIdx-1}.choice + 1;
    else
        prevChosenBandit = -1;
    end

    if (chosenBandit == 0) % Invalid trial. Skip.
        continue;
    end
    
    % Sampler
    for b = 1:numBandits
        bPrevIdxs   = reshape(combs{trialIdx, b}.', 1, []);
        rwdval{b}  = [choicerec(bPrevIdxs, 2)'];
        pval{b}    = [alpha .* ( (1-alpha).^(trialIdx-bPrevIdxs))];

        if (length(rwdval{b}) < 1)
            rwdval{b} = [0];
            pval{b}   = [1];
        end

        pval{b}    = pval{b}./sum(pval{b});
        rwdval{b} = sign(rwdval{b});
    end

    % TD
%     Q_td(trialIdx, :) = runQ;   % Save record of Q-values used to make TD-model based choice.
    rpe_td(trialIdx) = reward - runQ(chosenBandit);
    runQ(chosenBandit)  = runQ(chosenBandit) + alpha_td * rpe_td(trialIdx);

    nonChosenBandits = find((1:numBandits) ~= chosenBandit);
%         nonChosenBandits

    otherBandit1 = nonChosenBandits(1);
    otherBandit2 = nonChosenBandits(2);
    I1 = otherBandit1 == prevChosenBandit;
    I2 = otherBandit2 == prevChosenBandit;
    Ic = chosenBandit == prevChosenBandit;

    rvmat1 = arrayfun(@(x)(exp(beta_c.*(I1 - Ic) - beta_td .* (runQ(chosenBandit) - runQ(otherBandit1)) - beta.*(x - rwdval{otherBandit1}(:)'))), [rwdval{chosenBandit}(:)], 'UniformOutput', false);
    rvmat2 = arrayfun(@(x)(exp(beta_c.*(I2 - Ic) - beta_td .* (runQ(chosenBandit) - runQ(otherBandit2)) - beta.*(x - rwdval{otherBandit2}(:)'))), [rwdval{chosenBandit}(:)], 'UniformOutput', false);

    rvmat1 = [rvmat1{:}];
    rvmat2 = [rvmat2{:}];

%         rvmat1
%         rvmat2
    
    j = length(rwdval{chosenBandit});
    k = length(rwdval{nonChosenBandits(1)});
    l = length(rwdval{nonChosenBandits(2)});
    
%         jkl = [j, k, l]

    rvmat = arrayfun(@(i)(reshape(rvmat1(i*k+1:(i+1)*k) + rvmat2(i*l+1:(i+1)*l)', 1, [])'), (0:j-1), 'UniformOutput', false);
    rvmat = vertcat(rvmat{:});
    
%     rvmat

    pmat1  = arrayfun(@(x)(x.*pval{otherBandit2}(:)'), [pval{otherBandit1}(:)], 'UniformOutput', false);
    pmat1 = [pmat1{:}];
    pmat2  = arrayfun(@(x)(x.*pmat1(:)'), [pval{chosenBandit}(:)], 'UniformOutput', false);
    pmat2 = [pmat2{:}];

%     pmat2

    softmaxterm = 1./(1 + rvmat);
    pc(trialIdx) = max(sum(pmat2.*softmaxterm'), 1e-32);

end

% pc

nloglik = -sum(log(pc(choiceTrials)));

% nloglik

% add in the log prior probability of the parameters
nloglik = nloglik - log(flags.pp_alpha(alpha));
nloglik = nloglik - log(flags.pp_beta(beta));
nloglik = nloglik - log(flags.pp_betaC(beta_c));
nloglik = nloglik - log(flags.pp_alpha(alpha_td));
nloglik = nloglik - log(flags.pp_beta(beta_td));

end
