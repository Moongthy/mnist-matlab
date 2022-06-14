%%% learning.m 숫자 인식 분류기 %%%
clear;
fileID = fopen('trainimages.bin');
X = fread(fileID,[60000 784])/255;
%trainimages를 열어서 28*28픽셀의 60000개의 파일을 reshape하고 gray scale로 rescale
%RGB에서 범위가 [0 255]여서 255가 검정 0이 흰색
fclose(fileID);
fileID = fopen('trainlabels.bin');
T = fread(fileID);
%각 파일의 트레이닝할 라벨부착 ex) X(1,:)는 5이다. X(2,:)는 0이다.
fclose(fileID);
T(T==0)=10;
%matlab은 1부터니깐 0은 10으로 두고한다.
input_node  = 28.*28; %28*28픽셀로 되어있는 image를 input으로
hidden_node = 30; %hidden layer의 갯수
output_node = 10; %output layer의 갯수 0부터 9까지 총 10개
rand('seed',10); %아래의 random number를 저장하는 것이다.
w = 0.1*(2*rand(hidden_node, input_node+1)-1);
v = 0.1*(2*rand(output_node, hidden_node+1)-1);
%w와 v를 rand하게 하는데 그걸 rand('seed',10)으로 저장해두는 것이다.
%rand('seed',k)를 사용하면 저장되어있는 rand벡터 or 행렬을 가져올 수 있다.
% ex) rand('seed',1); rand(3)하면 3*3행렬이 뜬다 그다음 rand(3)을하면 다시 랜덤한 행렬이 뜬다.
%     이때 rand('seed',1);을 하고 rand(3)을하면 처음에 나온 랜덤행렬이 나온다.
eta=0.3; MaxIter=50;Iter=1;
Tol=1.0e-4; Resid = Tol*2;
E1=0;
for m=1:length(T) %60000개
    xl=[1 X(m,:)]';
    uh=w*xl; %hidden
    z=sigmoid(uh); %uh를 sigmoid로 푼다.
    z=[1;z]; %위에서 구한 z에서 앞에 1을 추가.
    uo=v*z; %output
    y=sigmoid(uo); %uo를 sigmoid로 푼다.
    t(1:output_node,1)=0; %0부터 1까지 숫자 정하기
    t(T(m))=1; %trainlabels할 숫자를 대입하기
    E1=E1+sum((y-t).^2); %output과 실제값의 에러구하기
end
E1=E1/length(T);
fprintf('%i-th update and error is %f\n ', Iter-1, E1); %0번째 한번해봤을때 에러 프린트
%여기선 일단 에러가 크게난다. 아래의 while문에서 트레이닝시작

%%
while Resid>=Tol && Iter < MaxIter
    %바로전스텝에서의 오차와 현재오차의 차이가 크게없거나 Iteration이 지정해둔 차례까지
    for m=1:length(T)
        xm=[1 X(m,:)]';
        uh=w*xm;
        z=sigmoid(uh);
        z=[1;z];
        uo=v*z;
        y=sigmoid(uo);
        t(1:output_node,1)=0;
        t(T(m))=1;
        del_k=d_sigmoid(uo).*(y-t);
        dEdv=del_k*z';
        del_j=d_sigmoid(uh).*(v(:,2:end)'*del_k);
        dEdw=del_j*xm';
        v=v-eta*dEdv;
        w=w-eta*dEdw;
    end
    E2=0;
    for m=1:length(T)
        xl=[1 X(m,:)]';
        uh=w*xl;
        z=sigmoid(uh);
        z=[1;z];
        uo=v*z;
        y=sigmoid(uo);
        t(1:output_node,1)=0;
        t(T(m))=1;
        E2=E2+sum((y-t).^2);
    end
    E2=E2/length(T);
    Resid = abs(E2-E1); %그전단계 에러와 차이가 거의 없으면 끝나게
    E1 = E2; %오차 에러
    fprintf('%i-th update and error is %f\n ',Iter, E1);
    Iter=Iter+1;
end
fprintf('The learning is finished \n');
save 'learningdata.mat'
