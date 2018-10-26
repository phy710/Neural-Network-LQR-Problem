function l = loss(d, y, opt)
    if strcmp(opt.loss, 'MSE') || strcmp(opt.loss, 'mse')
        l = mean((d(:)-y(:)).^2);
    elseif strcmp(opt.loss, 'cross-entropy') || strcmp(opt.loss, 'crossEntropy') || strcmp(opt.loss, 'cross entropy')
        l = -mean(d(:).*log(y(:)));
    else
        error('Loss function is not supported!');
    end
end