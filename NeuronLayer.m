classdef NeuronLayer
    properties
        W;
        b;
    end
    methods
        function obj = NeuronLayer(W_,b_)
            obj.W = W_;
            obj.b = b_;
            
        end
    end
end
