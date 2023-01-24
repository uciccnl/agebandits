param_to_plot = 'alpha'

analysis_constants;
paramNames = {'alpha', 'beta', 'beta_c'};
paramIdx = find(cellfun(@(x)(streq(x, param_to_plot)), {paramTable{:,1}}));

param(1).name = paramNames{1};
param(2).name = paramNames{2};
param(3).name = paramNames{3};

paramRange = paramTable{paramIdx,2}{:};
paramIdx
paramRange

param(paramIdx).lb = transform_params(paramRange(1)+0.05, {param(paramIdx).name}, 1);
param(paramIdx).ub = transform_params(paramRange(2)-0.05, {param(paramIdx).name}, 1);

param(paramIdx)

y = arrayfun(@(x)(unifrnd(param(paramIdx).lb, param(paramIdx).ub)), 0:10000, 'UniformOutput', false);
y = [y{:}];
ytr = arrayfun(@(x)(transform_params(x, {paramNames{paramIdx}})), y, 'UniformOutput', false);
ytr = [ytr{:}];

tiledlayout(2,1)
ax1 = nexttile;
hist(ax1, y)
ax2 = nexttile;
hist(ax2, ytr)