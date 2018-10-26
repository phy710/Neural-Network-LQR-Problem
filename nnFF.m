function layers = nnFF(layers, x)
    layers{1}.output_ = layers{1}.w*x + layers{1}.b;
    layers{1}.output = act(layers{1}.output_, layers{1}.act);
    if numel(layers) >= 2
        for a = 2 : numel(layers)
            layers{a}.output_ = layers{a}.w*layers{a-1}.output + layers{a}.b;
            layers{a}.output = act(layers{a}.output_, layers{a}.act);
        end
    end
end
