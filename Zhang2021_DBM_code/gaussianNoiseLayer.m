classdef gaussianNoiseLayer < nnet.layer.Layer
    % gaussianNoiseLayer   Gaussian noise layer
    %   A Gaussian noise layer adds random Gaussian noise to the input.
    %
    %   To create a Gaussian noise layer, use 
    %   layer = gaussianNoiseLayer(sigma, name)

    properties
        % Standard deviation.
        Sigma
    end
    
    methods
        function layer = gaussianNoiseLayer(sigma, name)
            % layer = gaussianNoiseLayer(sigma,name) creates a Gaussian
            % noise layer and specifies the standard deviation and layer
            % name.
            
            layer.Name = name;
            layer.Description = ...
                "Gaussian noise with standard deviation " + sigma;
            layer.Type = "Gaussian Noise";
            layer.Sigma = sigma;
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer for prediction and outputs the result Z.
            
            % At prediction time, the output is equal to the input.
            Z = X;
        end
        
        function [Z, memory] = forward(layer, X)
            % Z = forward(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            % At training time, the layer adds Gaussian noise to the input.
            sigma = layer.Sigma;
            noise = randn(size(X)) * sigma;
            Z = X + noise;
            
            memory = [];
        end
        
        function dLdX = backward(layer, X, Z, dLdZ, memory)
            % [dLdX, dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
            % backward propagates the derivative of the loss function
            % through the layer.
            
            % Since the layer adds a random constant, the derivative dLdX
            % is equal to dLdZ.
            dLdX = dLdZ;
        end
    end
end