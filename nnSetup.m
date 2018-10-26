function [layers, opt] = nnSetup()
    opt.eta = 0.001;
    opt.loss = 'MSE';
    opt.batchSize = 1;
    opt.L2 = 0.001;
    opt.shuffle = false;
    layers{1}.w = rand(64, 4)/5-0.1;
    layers{1}.act = 'ReLU';
    layers{2}.w = rand(64, size(layers{1}.w, 1))/5-0.1;
    layers{2}.act = 'ReLU'; 
    layers{3}.w = rand(64, size(layers{2}.w, 1))/5-0.1;
    layers{3}.act = 'ReLU'; 
    layers{4}.w = rand(18, size(layers{3}.w, 1))/5-0.1;
    layers{4}.act = 'custom';
    for a = 1 : numel(layers)
        layers{a}.b = rand(size(layers{a}.w, 1), 1)/5-0.1;
    end
end