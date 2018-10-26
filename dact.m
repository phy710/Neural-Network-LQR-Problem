function y = dact(x, activation)
    if strcmp(activation, 'none') || strcmp(activation, 'linear')
        y = ones(size(x));
    elseif strcmp(activation, 'ReLU') || strcmp(activation, 'relu') || strcmp(activation, 'reLU')
        y = ones(size(x));
        y(x<0) = 0;
        y(x==0) = 0.5;
    elseif strcmp(activation, 'sigmoid')
        y = exp(-x)./(1+exp(-x)).^2;
    elseif strcmp(activation, 'tanh')
        y = 1-tanh(x).^2;
    elseif strcmp(activation, 'custom')
        y = ones(size(x));
        y(x<-5) = 0;
        y(x>5) = 0;
    else
        error('Activation function is not supported!');
    end
end