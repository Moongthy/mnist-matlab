%%% MNIST database »Æ¿Œ %%%
clear;
fileID = fopen('trainimages.bin');
X = fread(fileID,[60000 784])/255;
fclose(fileID);
fileID = fopen('trainlabels.bin');
T = fread(fileID);
fclose(fileID);
T(T==0)=10;
image_number = 5;
visual(1:28,1:28)=0;
for i=1:28
    visual(i,1:28) = X(image_number,28*(i-1)+1:28*i);
end
imshow(visual)
T(image_number)