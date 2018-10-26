function [l, ME] = nnEvaluate(layers, opt, x, d)
    layers = nnFF(layers, x);
    y = layers{end}.output;
    l = loss(d, y, opt);
    ME = mean(abs((d(:)-y(:))));
end