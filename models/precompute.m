function precompute(sub, maxSamples)
% function precompute(sub, maxSamples)
%
% Precompute the subject choicerec and combs structures, up to maxSamples
%
% choicerec = [bandit rwdval isprobe]
%
%   memprobe trials contain the choice and rwdval of the probed trial.
%

mode = ''
dataDir = strcat('../transformed_Data_06.13.22_FILES', mode);
dataPattern = 'transformed_Data*.mat';

[sd, ~] = loadSubj(sub, dataDir, dataPattern);

numBandits = 3;
maxTrials = 180;

choicerec   = zeros(length(maxTrials), 3);
for numSamples = 1:maxSamples;
    % Precompute sets of all subsets
    for trialIdx = 1:maxTrials;
        if (trialIdx > 1)
            % Permutations: order matters as we want to count multiple draws each time
            combtmp = combinator(trialIdx-1, numSamples, 'p', 'r');

            % Now store choice and reward value or evoked choice and reward value for that trial
            % Mark evoked trials with 1 in the last column, in case the model wants to process them differently.
%             if (isempty(sd.trialrec{trialIdx}.bandits))
%                 % Is this a valid probe trial?  Then pull out the evoked bandit
%                 if (sd.trialrec{trialIdx}.choice == 1 && sd.trialrec{trialIdx}.rwdval>0)
%                     thisProbed = sd.trialrec{trialIdx}.probed;
%                     for searchIdx = 1:trialIdx-1;
%                         thisChoice = sd.trialrec{searchIdx}.choice;
%                         otherChoice = ~(thisChoice-1)+1;
%                         if (thisProbed == sd.trialrec{searchIdx}.probed)
%                             choicerec(trialIdx, :) = [sd.trialrec{searchIdx}.choice sd.trialrec{searchIdx}.rwdval 1];
%                             break;
%                         end
%                     end
%                 else
%                     % Invalid probe.  Won't be useful.  Mark it zero, dude.
%                     choicerec(trialIdx, :) = [0 0 1];
%                 end
%             else
                % For each trial, store choice and reward value
            choicerec(trialIdx, :) = [sd.trialrec{trialIdx}.choice sd.trialrec{trialIdx}.rwdval 0];
%             end

            for thisBandit = 1:numBandits;
                % Parse out invalid combinations - those that include more than one bandit type
                thisBanditTrials    = find(choicerec(1:trialIdx-1, 1)==thisBandit-1);
                if (numSamples == 1)
                    banditTmp   = combtmp(arrayfun(@(x)(all(ismember(x, thisBanditTrials))), combtmp)',:);
                else
                    banditTmp   = combtmp(all(arrayfun(@(x)(all(ismember(x, thisBanditTrials))), combtmp)')',:);
                end

                combs{numSamples}{trialIdx, thisBandit} = banditTmp;
            end

        else
            % Straight bandit trial.
            choicerec(trialIdx, :) = [sd.trialrec{trialIdx}.choice sd.trialrec{trialIdx}.rwdval 0];

        end % trialIdx > 1
    end

    datafn = ['precomputed/precomputed_sub' ...
               num2str(sub) '_' ...
               num2str(numSamples) '.mat'];
    save(datafn, 'combs', ...
                 'choicerec');

    disp(['Saved in ' datafn]);
end

