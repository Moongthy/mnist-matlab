%%% 검증 오차 %%%
function err=err(w,v)
fileID = fopen('testimages.bin');
X = fread(fileID,[10000 784])/255;
fclose(fileID);
fileID = fopen('testlabels.bin');
T = fread(fileID);
fclose(fileID);
T(T==0)=10;
testT(testT==0)=10;
N_Test=length(testT);
N_correct=0;
for k=1:N_Test
    test=[1 testX(k,:)]';
    z=sigmoid(w*test);
    z=[1;z];
    y=sigmoid(v*z);
    maxIndex=find(y == max(y));
    if maxIndex==testT(k)
        N_correct=N_correct+1;
    end
end
err = 1- N_correct/N_Test;