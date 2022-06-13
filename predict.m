%%% predict.m ���� ���� %%%
clear
load 'learningdata.mat'
imshow('test1.png');
newdata = imread('test1.png');
newdata = 1-double(newdata(:,:,1))/255;
newX = [];
for i=1:28
    newX = [newX newdata(i,:)];
end
newX=[1 newX]';
z=sigmoid(w*newX);
z=[1;z];
y=sigmoid(v*z);
predict_value=find(y == max(y));
if predict_value==10
    predict_value=0;
end
fprintf('New image is  : %d \n', predict_value);
