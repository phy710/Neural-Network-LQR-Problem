function layers = nnBP(layers, opt, x, d)
     if strcmp(opt.loss, 'MSE') || strcmp(opt.loss, 'mse')
         delta = (d-layers{end}.output).*dact(layers{end}.output_, layers{end}.act);
     elseif strcmp(opt.loss, 'cross-entropy') || strcmp(opt.loss, 'crossEntropy') || strcmp(opt.loss, 'cross entropy')
         if strcmp(layers{end}.act, 'softmax')
            delta = d-layers{end}.output;
         else
             error('Activation function of final layer should be softmax for cross-entropy loss function!');
         end
     else
         error('Loss function is not supported!');
     end
     for a = numel(layers) : -1 : 2
         layers{a}.w = layers{a}.w + opt.eta*delta*layers{a-1}.output'/opt.batchSize - opt.eta*opt.L2*layers{a}.w/opt.batchSize;
         layers{a}.b = layers{a}.b + mean(opt.eta*delta, 2) - opt.eta*opt.L2*layers{a}.b/opt.batchSize;
         delta = layers{a}.w'*delta.*dact(layers{a-1}.output_, layers{a-1}.act);
     end
     layers{1}.w = layers{1}.w + opt.eta*delta*x'/opt.batchSize - opt.eta*opt.L2*layers{1}.w/opt.batchSize;
     layers{1}.b = layers{1}.b + mean(opt.eta*delta, 2) - opt.eta*opt.L2*layers{1}.b/opt.batchSize;
end