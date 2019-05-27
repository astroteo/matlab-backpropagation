clear all 
close all
clc
%% Test settings

functional_test = 0;
binary_blobs_test = 0;
mnist_binary_test =1;
%% Learn about key-value pairs implementation in Matlab.

if functional_test 
    
n_input = 28 * 28;
n_hidden = 124;
n_output = 2;
W_output =  2 * rand([n_output,n_hidden]) - ones(n_output,n_hidden);
b_output = 2 * rand([n_output,1]) - ones(n_output,1);
W_hidden =  2 * rand([n_hidden, n_input]) - ones(n_hidden, n_input);
b_hidden = 2 * rand([n_hidden,1]) - ones(n_hidden,1);

%Layers = containers.Map({'W_output','b_output','W_hidden','b_hidden'},...
                         %{W_output,b_output,W_hidden, b_hidden});
                     
WL_ft = {W_hidden, W_output}
bL_ft = {b_hidden, b_output}

batch_size = 4;
X_FunctionalTest = 2 * rand(n_input, batch_size) -ones(n_input,batch_size);
Y_FunctionalTest = 1 * round(rand (n_output, batch_size))

                     
nn_ft = NeuralNetwork(WL_ft, bL_ft, 0.01, 10)

[A_ForwardPass, store_FunctionalTest] = nn_ft.forward(X_FunctionalTest)
derivatives_FunctionalTest = nn_ft.backward(X_FunctionalTest,Y_FunctionalTest, store_FunctionalTest)


nn_ft.fit(X_FunctionalTest, Y_FunctionalTest)

end
%% Binary Classifier simple test => binary cross entropy
if binary_blobs_test 
[~,~,data_train] = xlsread('C:\Users\BAMA306\toy2ClassesTrain.csv');
[~,~,data_test] = xlsread('C:\Users\BAMA306\toy2ClassesTest.csv');

data_train = cell2mat(data_train);
data_test = cell2mat(data_test);

X_train = data_train(:, 1:end-1);
Y_train = data_train(:,end);

Y_train = Y_train';
X_train = X_train';

n_input = 20;
n_hidden1 = 24;
n_hidden2 = n_hidden1;
n_output = 1;

W_hidden1 = 1/sqrt(n_input) * rand([n_hidden1, n_input]);  %2 * rand([n_hidden1, n_input]) - ones(n_hidden1, n_input);
b_hidden1 = zeros([n_hidden1,1]);% 2 * rand([n_hidden1,1]) - ones(n_hidden1,1);
W_hidden2 = 1/sqrt(n_hidden1) * rand([n_hidden2, n_hidden1]); %2 * rand([n_hidden2, n_hidden1]) - ones(n_hidden2, n_hidden1);
b_hidden2 = zeros([n_hidden2,1]);%2 * rand([n_hidden2,1]) - ones(n_hidden2,1);

W_output =  1/sqrt(n_hidden2) * rand([n_output,n_hidden2]);%2 * rand([n_output,n_hidden2]) - ones(n_output,n_hidden2);
b_output = zeros([n_output,1]);%2 * rand([n_output,1])- ones(n_output,1);


WL_ = {W_hidden1, W_output};
bL_ = {b_hidden1, b_output};

epochs_ = 15000;
lr_ = 0.001;

nn = NeuralNetwork(WL_, bL_, lr_, epochs_)

% test single iteration Fw/Bw pass:
[A_fp, store_fp] = nn.forward(X_train(:,1:100))
disp("OOKKK FWD")
derivatives_fp = nn.backward(X_train(:,1:100), Y_train(1:100) , store_fp)


% test over the first 20 samples of training dataset:
[Ap_t1,Yp_t1] = nn.predict(X_train(:,1:20))
Y_t1 = Y_train(1:20)


% test over the first 20 samples of training dataset  | BEFORE TRAINING:
[Ap_t1,Yp_t1] = nn.predict(X_train(:,1:20))
Y_t1 = Y_train(1:20)

[~,loss_plot, acc_plot] = nn.fit(X_train(:,1:5),Y_train(1:5));

% test over the first 20 samples of training dataset | AFTER TRAINING:
[Ap_t1,Yp_t1] = nn.predict(X_train(:,1:20))
Y_t1 = Y_train(1:20)

epochs_plot = 1 : 1 :epochs_;


figure()
plot(epochs_plot,loss_plot)
hold on

figure()
plot(epochs_plot,acc_plot)
hold on


% test over the test.set contained in the second file:
X_test= data_test(:, 1:end-1);
Y_test = data_test(:,end);

Y_test = Y_test';
X_test = X_test';

[Ap_t1,Yp] = nn.predict(X_test);

fid_out = fopen('file_out.csv','w');

result_check = zeros( max(size(Y_test)),n_input+2);

for i = 1: max(size(Y_test))
   result_check(i,:) = [data_test(i,:), Yp(i)];
    
end

csvwrite('C:\Users\BAMA306\toy2ClassesResults.csv',result_check)


end
%% MNIST BINARY CLASSIFIER 
if mnist_binary_test
img_file ='C:\Users\BAMA306\train-images.idx3-ubyte' 
label_file = 'C:\Users\BAMA306\train-labels.idx1-ubyte'

[imgs,labels] = readMNIST(img_file, label_file, [20,20], 0);

visualize_first_images = 0
eval_nr = 1

if visualize_first_images
    for eval_nr = 1:5
        figure()
        title("expected label is: "+ num2str(labels(eval_nr)))
        image([0,200],[0,200],imgs(:,:,eval_nr)*255)
        hold on
    end
end


step = 200;
offset = 0;

Xmnist = zeros(400, 10000);
Ymnist = zeros(1,10000);

for i = 1 : step: 10000
    
    [imgs,labels] = readMNIST(img_file, label_file, [step,step], 0);
    offset = offset + step;
    
    for j = 1:step
        img =imgs(:,:,j);
        img = img(:);
        Xmnist(:,i+j) = img;
        Ymnist(i+j) = labels(j);
    end
    
end


Xminst = Xmnist(:,1:10000);
Ymnist = Ymnist(:,1:10000);


cbi = 1;
i = 1;
for y = Ymnist
    if y == 5 || y == 8
       
        XbinaryMnist(:, cbi) = Xmnist(:,i);
        
        if y ==5
            YbinaryMnist(:, cbi) = 0;
        else
             YbinaryMnist(:, cbi) = 1;
        end
        cbi = cbi +1 ;
        
    end
    i = i+1;
end

XbinaryMnist_train  = XbinaryMnist(:,1:floor(max(size(XbinaryMnist))*0.8));
YbinaryMnist_train = YbinaryMnist(:,1:floor(max(size(XbinaryMnist))*0.8));

XbinaryMnist_test  = XbinaryMnist(:,floor(max(size(XbinaryMnist))* 0.8) +1 : end);
YbinaryMnist_test =  YbinaryMnist(:,floor(max(size(XbinaryMnist))* 0.8) +1 :end);

bp = "C:\Users\BAMA306\Documents\Datascience\";
%XYbinaryMnist_train = [XbinaryMnist_train,YbinaryMnist_train];
csvwrite(bp + "Xbinary_train.csv",XbinaryMnist_train)
csvwrite(bp + "Ybinary_train.csv",YbinaryMnist_train)
csvwrite(bp + "Xbinary_test.csv",XbinaryMnist_train)
csvwrite(bp + "Ybinary_test.csv",YbinaryMnist_train)




n_input = 20 * 20;%padding removed authomatically
n_hidden1 = 196;
n_hidden2 = 10;
n_output = 1;

W_hidden1 = 1/sqrt(n_input)* rand([n_hidden1, n_input]);% 1/sqrt(n_input) * ;  %2 * rand([n_hidden1, n_input]) - ones(n_hidden1, n_input);
b_hidden1 = zeros([n_hidden1,1]);% 2 * rand([n_hidden1,1]) - ones(n_hidden1,1);
W_hidden2 = 1/sqrt(n_hidden1) * rand([n_hidden2, n_hidden1]); %1/sqrt(n_hidden1) * rand([n_hidden2, n_hidden1]); %2 * rand([n_hidden2, n_hidden1]) - ones(n_hidden2, n_hidden1);
b_hidden2 = zeros([n_hidden2,1]);%2 * rand([n_hidden2,1]) - ones(n_hidden2,1);

W_output =  1/sqrt(n_hidden2) * rand([n_output,n_hidden2]) ; %1/sqrt(n_hidden2) * rand([n_output,n_hidden2]);%2 * rand([n_output,n_hidden2]) - ones(n_output,n_hidden2);
b_output = zeros([n_output,1]);%2 * rand([n_output,1])- ones(n_output,1);


%WL_ = {W_hidden1,W_hidden2, W_output};
%bL_ = {b_hidden1,b_hidden2, b_output};

W_shidden =  1/sqrt(n_input) * rand([n_hidden1, n_input]);% 1/sqrt(n_input) * ;  %2 * rand([n_hidden1, n_input]) - ones(n_hidden1, n_input);
b_shidden = zeros([n_hidden1,1]);% 2 * rand([n_hidden1,1]) - ones(n_hidden1,1);

W_soutput = 1/sqrt(n_hidden1)* rand([n_output, n_hidden1]);% 1/sqrt(n_input) * ;  %2 * rand([n_hidden1, n_input]) - ones(n_hidden1, n_input);
b_soutput = zeros([n_output,1]);% 2 * rand([n_hidden1,1]) - ones(n_hidden1,1);

          %=> THIS TYPE OF INITIALIZATION IS SUPER-TOP WITH tanh activation
          
%WL_ = {W_shidden, W_soutput};
%bL_ = {b_shidden, b_soutput};
WL_ = {W_hidden1,W_hidden2, W_output};
bL_ = {b_hidden1,b_hidden2, b_output};

epochs_ = 15000;
lr_ = 0.01;

mnist_net =  NeuralNetwork(WL_, bL_, lr_, epochs_);

%Functional Tests:
[Ap_t1,Yp_t1] = mnist_net.predict(XbinaryMnist_train(:,1:20))
[Ap_f,store_ft] = mnist_net.forward(XbinaryMnist_train(:,1:2))
derivatives_ft = mnist_net.backward(XbinaryMnist_train(:,1:2),YbinaryMnist_train(1:2),store_ft)
YbinaryMnist_train(:,1:20)

%Train
[~,loss_plot_mnist, acc_plot_mnist] = mnist_net.fit(XbinaryMnist_train, YbinaryMnist_train)

epochs_plot = 1 : 1 :epochs_;

figure()
plot(epochs_plot,loss_plot_mnist)
hold on

figure()
plot(epochs_plot,acc_plot_mnist)
hold on

mnist_net.eval(XbinaryMnist_test, YbinaryMnist_test)
mnist_net.verbose =0;
offset =0
 for eval_nr = 1:100
        [imgs,labels] = readMNIST(img_file, label_file, [2,2], offset);
        for i =1:2
            if labels(i) == 5 || labels(i) ==8
                true_lab = labels(i);
                img = imgs(:,:,i);
                img = img(:);
                [prob,lab] = mnist_net.predict(img);
                if lab == 1
                  lab = 8;
                else
                  lab = 5;
                  prob = 1-prob; % probability that this is not 8
                end
        
        figure()
        title("expected label is: "+ num2str(true_lab) + "predicted label is: " + num2str(lab) +"with probability: "+ num2str(prob))
        hold on
        image([0,200],[0,200],imgs(:,:,i)*255)
        hold on
                  
                  
            end
        end
        offset = offset +2;
       
 end
    

 
 
end
%% MNIST experiment 10 labels classifier.


