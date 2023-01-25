function [sd, bv] = loadSubj(subj, datadir, datapattern)
% function [sd bv] = loadSubj(subj, [datadir='.'], [datapattern='*.mat'])
%

if (nargin < 2)
    datadir = '.';
end

if (nargin < 3)
    datapattern = '*.mat';
end

dd = dir(fullfile(datadir, datapattern));
sd = load(fullfile(datadir, dd(subj).name));
disp([dd(subj).name]);
bv  = {};
% if (0)
%     bvdir = [datadir 'badvols/'];
%     dbv = dir(fullfile(bvdir, ['badvols_' ...
%                                num2str(subj) '_' datapattern]));
%     for bvi = 1:length(dbv);
%         bv{bvi} = load(fullfile(bvdir,dbv(bvi).name));
%     end
% end
