clear;
clc;
close all;
load data.mat;
xx = X(:, 1:100);
dd = U(:, 1:100);
x = X(:, 101:1100);
d = U(:, 101:1100);
epochMax = 3000;
etaDecay = 0.95;
rand('seed', 0);
[layers, opt] = nnSetup();
[trainLoss, trainME] = nnEvaluate(layers, opt, x, d);
[testLoss, testME] = nnEvaluate(layers, opt, xx, dd);
tic;
epoch = 0;
while trainLoss(end)>0 && epoch<epochMax
    epoch = epoch+1;
    layers = nnTrain(layers, opt, x, d);
    [trainLoss(epoch+1), trainME(epoch+1)]= nnEvaluate(layers, opt, x, d);
    [testLoss(epoch+1), testME(epoch+1)]= nnEvaluate(layers, opt, xx, dd);
    disp(['Epoch: ' num2str(epoch) ', Learning rate: ' num2str(opt.eta) ', Loss: ' num2str(trainLoss(end)) '/' num2str(testLoss(end)) ', ME: ' num2str(trainME(end)) '/' num2str(testME(end))]);
    if trainLoss(epoch+1)>trainLoss(epoch)
        % If loss increases, decrease learning rate
        opt.eta = opt.eta*etaDecay;
    end
end
toc;
disp(['Training loss: ' num2str(trainLoss(end))]);
disp(['Test loss: ' num2str(testLoss(end))]);
disp(['Mean of desired output on the test data: ' num2str(mean(dd(:)))]);
disp(['Std of desired output on the test data: ' num2str(std(dd(:)))]);
disp(['Training mean error: ' num2str(trainME(end))]);
disp(['Test mean error: ' num2str(testME(end))]);
figure;
plot(0:epoch, trainLoss);
hold on;
plot(0:epoch, testLoss);
grid on;
title('Loss');
xlabel('Epoch'); 
ylabel('Mean Squared Error');
legend('Training', 'Test');
figure;
plot(0:epoch, trainME);
hold on;
plot(0:epoch, testME);
grid on;
title('Mean Error');
xlabel('Epoch');
ylabel('Mean Error');
legend('Training', 'Test');
figure;
semilogy(0:epoch, trainLoss);
hold on;
semilogy(0:epoch, testLoss);
grid on;
title('Loss');
xlabel('Epoch'); 
ylabel('Mean Squared Error (log)');
legend('Training', 'Test');
disp(['Training mean error: ' num2str(trainME(end))]);
disp(['Test mean error: ' num2str(testME(end))]);
figure;
semilogy(0:epoch, trainME);
hold on;
semilogy(0:epoch, testME);
grid on;
title('Mean Error');
xlabel('Epoch');
ylabel('Mean Error (log)');
legend('Training', 'Test');

save layers.mat layers trainLoss testLoss trainME testME