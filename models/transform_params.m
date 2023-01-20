function transformedParams = transformParams(params, paramNames, direction)
% function transformedParams = transformParams(params, paramNames, [direction])

analysis_constants;

transformedParams = params;

if nargin < 3
    direction = 0; % IN
end

if (streq(paramNames{1}, ''))
    return;
end

% paramTable
% paramNames
for paramIdx = 1:length(params);
%     paramIdx
    transformerIndex = find(cellfun(@(x)(streq(x, paramNames{paramIdx})), {paramTable{:,1}}));
    paramRange = paramTable{transformerIndex,2}{:};

    if (direction == 0)
        % min + [max-min]./[1+exp(-x)]
        transformedParams(paramIdx) = paramRange(1) + (paramRange(2)-paramRange(1))./[1+exp(-params(paramIdx))];
    else
        transformedParams(paramIdx) = -log(-1 + [paramRange(2)-paramRange(1)]./[params(paramIdx)-paramRange(1)]);
    end
end
