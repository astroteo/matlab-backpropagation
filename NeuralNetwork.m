classdef NeuralNetwork
    properties
        WL;
        bL;
        L;
        parameters;
        n;%samples
        verbose = 1;
        lr;
        loss_type ="binary_cross_entropy"
        epochs;
            %TODO: 
            %    add classes names(as a possibility).
            %    add batch_size training in costructor
            %    add all desired loss types
            
    end
    
    methods %constructor & setters/getteres
        function obj = NeuralNetwork(WL_, bL_, lr_, epochs_,loss_type_)
            
            obj.WL = WL_;
            obj.bL = bL_;
            
            obj.L = max(size(WL_));
            
            obj.parameters = containers.Map;
            cw =1;
            for wL = WL_
                obj.parameters("W"+num2str(cw)) = cell2mat(wL);
                cw = cw+1;
            end
            
            cw = 1;
             for bL_ = bL_
                obj.parameters("b"+num2str(cw)) = cell2mat(bL_);
                cw = cw+1;
             end
            
            disp("L: " + num2str(obj.L))
            obj.n =0;
            disp("n: "+ num2str(obj.n))
            
            obj.lr = lr_;
            obj.epochs = epochs_;
            
            
            
            if nargin == 5
                obj.loss_type = loss_type_;
            end
        
        end
        
        function obj = set_verbose(obj,verbose_)
            obj.verbose = verbose_;
        end
        function obj = set_learningRate(obj, lr_)
            obj.lr = lr_;
        end
        
    end
        
    methods %functions and attributes
        function Z = sigmoid_(obj,X)
            
           Z = 1./ (1+ exp(-X));
           
        end
        
        function dZ = sigmoid_derivative(obj,X)
            
            s = obj.sigmoid_(X);
            
           dZ = s .*( 1 - s);
           
        end
        
        function Z = tanh_(obj,X)
            Z = tanh(X);
        end
        
        function dZ = tanh_derivative(obj,X)
            dZ = sech(X).^2;
        end

        function [A,store] = forward(obj,X)
            
            store = containers.Map;
            A = X;
            
            obj.n = max(size(X(1,:)));
  
            for l = 1 :1: obj.L-1
                
                if obj.verbose
                    disp("........................................................")
                    disp("layer: " + num2str(l))
                    disp("W-size: "+ num2str(size(obj.parameters("W"+num2str(l)))))
                    disp("b-size:" + num2str(size(obj.parameters("b"+num2str(l)))))
                    disp("A-size:" + num2str(size(A)))
                    
%                     disp("A_prev =>"+num2str(A))
%                     Z_disp = obj.parameters("W"+num2str(l))* obj.sigmoid_(A) + ...
%                                                     obj.parameters("b"+num2str(l));
%                     disp("Z =>"+num2str(Z_disp))
%                     
%                      A_disp = obj.sigmoid_(Z_disp);
%                     
%                     disp("A_next => " + num2str(A_disp))
%                     
                    disp("........................................................")
                end
              
                Z = obj.parameters("W"+num2str(l))* A + ...
                                                    obj.parameters("b"+num2str(l));
                %Z = obj.parameters("W"+num2str(l))* A + ...
                                                    %obj.parameters("b"+num2str(l));
                A = obj.sigmoid_(Z);
                %A = Z;
                %A = obj.tanh_(Z)
              
                        %TODO: replace with list of activation functions =>
                        %   obj.activation(...)
                store("A"+num2str(l)) = A;
                store("W"+num2str(l)) = obj.parameters("W"+num2str(l));
                store("Z"+num2str(l)) = Z;
                
               
                        
                
                
            end
            
            
            Z = obj.parameters("W"+num2str(obj.L))* A + ...
                                        obj.parameters("b"+num2str(obj.L));
            A = obj.sigmoid_(Z);
                        %TODO: subsitutite with other oput loss (include
                        %                                       regression)
            store("A"+num2str(obj.L)) = A;
            store("Z"+num2str(obj.L)) = Z;
            store("W"+num2str(obj.L)) = obj.parameters("W"+num2str(obj.L));
            
            if obj.verbose
                
                disp("******************************************************")
                
                disp("layer: " + num2str(obj.L) + " => output")
                disp("W-size: "+ num2str(size(obj.parameters("W"+num2str(obj.L)))))
                disp("b-size:" + num2str(size(obj.parameters("b"+num2str(obj.L)))))
                disp("A-size:" + num2str(size(A)))
                disp("Z-size:" + num2str(size(Z)))
                
                disp("********************************************************")
           
                disp("stored variables from forward pass: ")
                disp(store.keys)
                disp("******************************************************")
                    
     
           end
           
            
            
            
        end
        
        function [derivatives] = backward(obj,X,Y,store)
            
            obj.n = max(size(X(1,:)));
            store("A0") = X;
         
            derivatives = containers.Map;
            A = store("A"+num2str(obj.L));
            
            dA = -1 * Y./A + (1 -Y) ./ (1 - A);
                        % TODO: subsititute with other losses
                        %      -> generic cross-entropy (multi-label)                    
                        %      -> L2-loss for regression
            
           
            
            dZ = dA .* obj.sigmoid_derivative(store("Z"+ num2str(obj.L)));%Hadamard product
            dW = 1/ obj.n * dZ * store("A" +num2str(obj.L -1))' ; 
            db = 1/obj.n * sum(dZ,2);% average db => sum over saples( rows)
            
            dAPrev = store("W"+num2str(obj.L))' * dZ;
            
            derivatives("dW"+num2str(obj.L)) = dW;
            derivatives("db"+num2str(obj.L)) = db;
            
              if obj.verbose
                disp("****************************************************** Backward layer #-"+ num2str(obj.L))
               
                disp(size(dA))
                disp(size(obj.sigmoid_derivative(store("Z"+ num2str(obj.L)))))

                disp("dW  size=>")
                disp(size(dW))
                
                disp("db size=>")
                disp(size(db))

                disp("dZ size =>")
                disp(size(dZ))
                disp(size( store("W"+num2str(obj.L))))

                disp("dA size =>")
                disp(size(dA))
                 
                disp("dAPrev size =>")
                disp(size(dAPrev))
                
                disp("samples n=>" + num2str(obj.n))
                
                disp("******************************************************") 
              end
            
            
            
            for l = obj.L-1:-1:1
                dZ = dAPrev .* obj.sigmoid_derivative(store("Z"+num2str(l)));
                %dZ = dAPrev .* store("Z"+num2str(l));
                                % TODO: subsititute with CORRECT activation
                                %      -> tanh,tanh_derivative                    
                dW = 1/obj.n * (dZ * store("A" + num2str(l-1))');
                db = 1/obj.n * sum(dZ,2);% avg db => sum over samples(rows)
                
                derivatives("dW"+num2str(l)) = dW;
                derivatives("db"+num2str(l)) = db;     
                
            
                if l > 1
                    dAPrev = store("W" + num2str(l))' * dZ;
                end
                     % TODO: subsititute with other activation function
               
                if obj.verbose
                    disp("****************************************************** Backward: Layer #-" + num2str(l))
                    
                    disp("dA  size=>")
                    disp(size(dA))
                    disp(size(obj.sigmoid_derivative(store("Z"+ num2str(obj.L)))))

                    disp("dW  size=>")
                    disp(size(dW))

                    disp("db size=>")
                    disp(size(db))

                    disp("dZ size =>")
                    disp(size(dZ))
                    
                    disp("dA size =>")
                    disp(size(dA))
                 
                    disp("dAPrev size =>")
                    disp(size(dAPrev))


                    disp("samples n=>" + num2str(obj.n))

             
                    disp("******************************************************")
                end
                
            end
             
        end
        
        function [Ap,Yp] = predict(obj,X)
             [Ap,~] = obj.forward(X); 
 
            [n_output,~] = size(obj.parameters("W"+num2str(obj.L)));
            Yp = zeros(size(Ap));
         
            if n_output ==1 
                
                
                if obj.loss_type == "binary_cross_entropy"
                     for i = 1: max(size(Ap))
                       yi = Ap(i);
                       if yi > 0.5
                           Yp(i) = 1;
                       end 
                           
                   end
                   
                   if obj.verbose
                    disp("----------------------------------")
                    disp("predicting with 0.5 thershold on probability (cross entropy loss)") 
                    disp(Yp)
                    disp("----------------------------------")
                    
                    end
                    
                    
                else
                    %TODO: implement regressor.
                    disp("no-regressor-modeled yet")
                end
                
                
            else
                Yp = max(Ap) - 1;
           end
            
        end
        
        function cost = loss(obj,Ap,Y)
            
             cost =0;
            if obj.loss_type == "binary_cross_entropy"
               
                for i=1:size(Y)
                    cost = cost + -1 * ( Y(i) * log(Ap(i)) + (1-Y(i)) * log(1-Ap(i)));
                end
            else
             
             for i = 1:size(Ap(1,:))
                cost = cost  -1 * ( Y(:,i)' * log(Ap(:,i)));   
             end
                %TODO: implement regressor cost, etc..
                     % !! AT THIS STAGE IS INCOHERENT (only binary c.e. supported)
            end
                
     
        end
        
        function acc = batch_accuracy(obj,Yp,Y)
             acc = 0;
             n_samples = max(size(Y));
             for i = 1 : n_samples
                 
                 if Yp(i) == Y(i)
                     acc = acc +1;
                 end
             end
             

        if n_samples > 0
            acc = acc / n_samples * 100;
        end
             
        end
        
        function  [obj, loss_vector, acc_vector] = fit(obj, X,Y, Xval, Yval)
            
            obj.verbose = 0;
                    % shut-up when training 
                    
             if nargin < 5
                 disp("validating using 20% of the training dataset")
                 n_samples = size(X(1,:));
                 n_samples = max(n_samples);
                 n_val = round(0.2 * n_samples);
                 Xval = X(:, (n_samples-n_val):n_samples);
                 Yval = Y((n_samples-n_val):n_samples);
                 
                 X = X(:,1:(n_samples-n_val-1));
                 Y = Y(:,1:(n_samples-n_val-1));
                    %TODO: replace with random sampling along the way.
             end
                    
             loss_vector = zeros(1,obj.epochs);
             acc_vector = zeros(1,obj.epochs);
            
            disp(" training started ------------------------------------")
            for epoch = 1 : obj.epochs
                
                %forward
                [~, store] = obj.forward(X);
                derivatives = obj.backward(X,Y,store);
                
               
                % update params
                for l = 1:1:obj.L
                     %disp("layer =>" + num2str(l))
                     %derivatives("dW"+num2str(l))
                    
                   
                    obj.parameters("W"+num2str(l)) = obj.parameters("W"+num2str(l)) - ...
                                                     obj.lr * derivatives("dW"+num2str(l));
                                                 
                    obj.parameters("b"+num2str(l)) = obj.parameters("b"+num2str(l)) -...
                                                     obj.lr  * derivatives("db"+num2str(l));
                    
                end
                
                
                 % score and evaluate
                 [Ap,Yp] = obj.predict(Xval);
                 curr_loss = obj.loss(Ap,Yval);
                 curr_acc = obj.batch_accuracy(Yp,Yval);
                 
                 loss_vector(1,epoch)= curr_loss;
                 acc_vector(1,epoch) = curr_acc;
                 
                 
                % display current results
                if epoch - round(epoch /1000) * 1000 == 0
                    
                    disp("epoch: " + num2str(epoch) + "/" + num2str(obj.epochs))
                    disp("  current loss => " + num2str(curr_loss))
                    disp("  current accuracy => " + num2str(curr_acc)+" % ")
                    
                end
                
                
               
             
                
            end
            
            disp(" -------------------------------------- training ended")
            
        end
        
        
        function  [loss, acc] = eval(obj, Xtest, Ytest)
            %predict and score
             [Ap,Yp] = obj.predict(Xtest);
             loss = obj.loss(Ap,Ytest);
             acc = obj.batch_accuracy(Yp,Ytest);
         
                 
                 
             % display current results
             disp(" -$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$: result")
             disp("resulting loss: " + num2str(loss))
             disp("resulting accuracy:" + num2str(acc))
                
             disp(" -$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$-$: result")
                
       end
            
          
        
        
        
      
    end
end
