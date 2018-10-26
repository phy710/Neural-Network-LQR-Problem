function y = act(x, activation)
    if strcmp(activation, 'none') || strcmp(activation, 'linear')
        y = x;
    elseif strcmp(activation, 'ReLU') || strcmp(activation, 'relu') || strcmp(activation, 'reLU')
        y = max(x, 0);
    elseif strcmp(activation, 'sigmoid')
        y = 1./(1+exp(-x));
    elseif strcmp(activation, 'tanh')
        y = tanh(x);
    elseif strcmp(activation, 'softmax')
        y = exp(x)./sum(exp(x));
    elseif strcmp(activation, 'custom')
        y = x;
        y(x>5) = 5;
        y(x<-5) = -5;
    else
        error('Activation function is not supported!');
    end
end