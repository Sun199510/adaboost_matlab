% 训练样本数据 

for i=1:1:10
    for j=1:1:53
        X3(i,j)=fft_ad_test1(frequency_select(r(j),1),frequency_select(r(j),1),i);
    end
end
train_fft = [train_fft X(:,end)];
train_cpl =[X(:,r(1:18)) X(:,end)];
train_flo = [X(:,r(1:33)) X(:,end)];
train_mic = [feature(:,r(1:9)) train_fft(:,end)];
test_fft=[X1;X2;X3];
test_cpl = [cpl_hc_test(:,r(1:18));cpl_hc_test(:,r(1:18));cpl_hc_test(:,r(1:18))];
test_flo = [flow_hc(:,r(1:33));flow_mci(:,r(1:33)) ;flow_ad(:,r(1:33))];
test_mic = [feature_hc(:,r(1:9));feature_mci(:,r(1:9));feature_ad(:,r(1:9))];
randIndex = randperm(size(train_fft,1));
% train_new = X(randIndex,:);
order = zeros(1,4);
%% ***********************************初试过程************************************  
H1=zeros(432,1);H2=H1;H3=H1;H4=H1;
for i=1:1:432 
    [H1(i),~,~] =svmpredict(train_fft(i,end),train_fft(i,1:end-1), model_fft);
    [H2(i),~,~] =svmpredict(train_cpl(i,end),train_cpl(i,1:end-1), model_cpl);
    [H3(i),~,~] =svmpredict(train_flo(i,end),train_flo(i,1:end-1), model_flo);
    [H4(i),~,~] =svmpredict(train_mic(i,end),train_mic(i,1:end-1), model_mic);
end  
errDataH1=find(H1~=train_fft(:,end));%找到被h1错分的样本点的序号  
errDataH2=find(H2~=train_fft(:,end));%找到被h2错分的样本点的序号  
errDataH3=find(H3~=train_fft(:,end));%找到被h3错分的样本点的序号
errDataH4=find(H4~=train_fft(:,end));%找到被h4错分的样本点的序号
accDataH1=find(H1==train_fft(:,end));%找到被h1正确分的样本点的序号  
accDataH2=find(H2==train_fft(:,end));%找到被h2正确分的样本点的序号  
accDataH3=find(H3==train_fft(:,end));%找到被h3正确分的样本点的序号 
accDataH4=find(H4==train_fft(:,end));%找到被h4正确分的样本点的序号 
errDataAll={errDataH1;errDataH2;errDataH3};  
accDataAll={accDataH1;accDataH2;accDataH3};  
  
N=432;  
D1=zeros(N,1)+1/N;       % 初始化权值分布  
%% ***********************************第一次迭代***********************************  
err1=sum(D1(errDataH1,:));%所有被错分类的样本点的权值之和即为误差率  
err2=sum(D1(errDataH2,:));%所有被错分类的样本点的权值之和即为误差率  
err3=sum(D1(errDataH3,:));%所有被错分类的样本点的权值之和即为误差率
err4=sum(D1(errDataH4,:));%所有被错分类的样本点的权值之和即为误差率

errAll=[err1,err2,err3,err4];  
[minErr,minIndex]=min(errAll);  
%根据误差率e1计算H1的系数：  
a1=0.5*log((2-minErr)/minErr);  
minErrData=errDataAll{minIndex};  
minAccData=accDataAll{minIndex};  
D2=D1;  
for i=minAccData'  
    D2(i)=D2(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D2(i)=D2(i)/(2*minErr);  
end  
  
%分类函数  
f1=a1.*H1;
order(minIndex)=a1;
% kindFinal=sign(f1)%此时强分类器的分类结果  
  
%% ***********************************第二次迭代***********************************  
err1=sum(D2(errDataH1,:));%所有被h1错分类的样本点的权值之和即为误差率  
err2=sum(D2(errDataH2,:));%所有被h2错分类的样本点的权值之和即为误差率  
err3=sum(D2(errDataH3,:));%所有被h3错分类的样本点的权值之和即为误差率
err4=sum(D2(errDataH4,:));%所有被h4错分类的样本点的权值之和即为误差率

errAll=[err1,err2,err3,err4];  
[minErr,minIndex]=min(errAll);  
% 根据误差率e2计算H2的系数：  
a2=0.5*log((2-minErr)/minErr);  
minErrData=errDataAll{minIndex};  
minAccData=accDataAll{minIndex};  
D3=D2;  
for i=minAccData'  
    D3(i)=D3(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D3(i)=D3(i)/(2*minErr);  
end  
% 分类函数
order(minIndex)=a2;
% f2=a1.*H1+a2*H2;  
% kindFinal=sign(f2)%此时强分类器的分类结果  
  
%% ***********************************第三次迭代***********************************  
err1=sum(D3(errDataH1,:));%所有被错分类的样本点的权值之和即为误差率  
err2=sum(D3(errDataH2,:));%所有被错分类的样本点的权值之和即为误差率  
err3=sum(D3(errDataH3,:));%所有被错分类的样本点的权值之和即为误差率  
err4=sum(D3(errDataH4,:));
errAll=[err1,err2,err3,err4];  
[minErr,minIndex]=min(errAll);  
minIndex=2;minErr=errAll(minIndex);
% 根据误差率e3计算G3的系数：  
a3=0.5*log((2-minErr)/minErr);  
minErrData=errDataAll{minIndex};  
minAccData=accDataAll{minIndex};  
D4=D3;  
for i=minAccData'  
    D4(i)=D4(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D4(i)=D4(i)/(2*minErr);  
end  
 
% 分类函数 
order(minIndex)=a3;
% f3=a1.*H1+a2*H2+a3*H3;  
% kindFinal=sign(f3)%此时强分类器的分类结果  
%%%%%%%%%%%%%%%%%%计算第四个弱分类器的权重%%%%%%%%%%%%%%%
err1=sum(D4(errDataH1,:));%所有被错分类的样本点的权值之和即为误差率  
err2=sum(D4(errDataH2,:));%所有被错分类的样本点的权值之和即为误差率  
err3=sum(D4(errDataH3,:));%所有被错分类的样本点的权值之和即为误差率  
err4=sum(D4(errDataH4,:));
errAll=[err1,err2,err3,err4];  
[minErr,minIndex]=min(errAll);  
minIndex=4;minErr=errAll(minIndex);
% 根据误差率e3计算G3的系数：  
a4=0.5*log((2-minErr)/minErr);  
minErrData=errDataAll{minIndex};  
minAccData=accDataAll{minIndex};  
D5=D4;  
for i=minAccData'  
    D4(i)=D4(i)/(2*(1-minErr));  
end  
for i=minErrData'  
     D4(i)=D4(i)/(2*minErr);  
end  
 
% 分类函数 
order(minIndex)=a4;
order1=order/sum(order);

final=order1(1)*H1+order1(2)*H2+order1(3)*H3+order1(4)*H4;
final1 = round(final);
length(find(final1~=train_fft(:,end)))/432;

for i=1:1:30
[H1(i),~,~] =svmpredict(test_fft(i,end),test_fft(i,1:end-1), model_fft);
[H2(i),~,~] =svmpredict(test_cpl(i,end),test_cpl(i,1:end-1), model_cpl);
[H3(i),~,~] =svmpredict(test_flo(i,end),test_flo(i,1:end-1), model_flo);
[H4(i),~,~] =svmpredict(test_mic(i,end),test_mic(i,1:end-1), model_mic);
end

%%%%%%%%%%制作测试集%%%%%%%%%%%
info1 = dir('D:\SJN\0715mat\process_new\test\hc');
info2 = dir('D:\SJN\0715mat\process_new\test\mci');
info3 = dir('D:\SJN\0715mat\process_new\test\ad');

for i=3:1:12
    load([info1(i).folder,'\',info1(i).name])
     fft_hc_test(:,:,i-2) = EEG.fft_result(:,1:81);
     cpl_hc_test(i,:) = EEG.lz_result;
     temp = EEG.DTF_sig;EEG.DTF_sig;
     theta = sum(temp(:,:,1:6),3);
     alpha = sum(temp(:,:,7:12),3);
     beta = sum(temp(:,:,13:19),3);
     gamma = sum(temp(:,:,20:29),3);
    flow_hc(i-2,:)= [reshape(theta',1,54*54) reshape(alpha',1,54*54) reshape(beta',1,54*54) reshape(gamma',1,54*54)];
     temp =mystand(EEG.EEG_microstates);
     for j=1:1:2000
         for k=1:1:20
         dis(k)=dist_E(temp(:,j),C(k,:)');
         end
         [~,state(j)]=min(dis);
     end
     feature_hc(i-2,:) = ts_analysis(state);
%     state =[state EEG.EEG_microstates];
end

for i=3:1:12
    load([info2(i).folder,'\',info2(i).name])
     fft_mci_test(:,:,i-2) = EEG.fft_result(:,1:81);
     cpl_mci_test(i,:) = EEG.lz_result;
     temp = EEG.DTF_sig;EEG.DTF_sig;
     theta = sum(temp(:,:,1:6),3);
     alpha = sum(temp(:,:,7:12),3);
     beta = sum(temp(:,:,13:19),3);
     gamma = sum(temp(:,:,20:29),3);
    flow_mci(i-2,:)= [reshape(theta',1,54*54) reshape(alpha',1,54*54) reshape(beta',1,54*54) reshape(gamma',1,54*54)];
     temp =mystand(EEG.EEG_microstates);
     for j=1:1:2000
         for k=1:1:20
         dis(k)=dist_E(temp(:,j),C(k,:)');
         end
         [~,state(j)]=min(dis);
     end
     feature_mci(i-2,:) = ts_analysis(state);
%     state =[state EEG.EEG_microstates];
end

for i=3:1:12
    load([info3(i).folder,'\',info3(i).name])
     fft_ad_test(:,:,i-2) = EEG.fft_result(:,1:81);
     cpl_ad_test(i,:) = EEG.lz_result;
     temp = EEG.DTF_sig;EEG.DTF_sig;
     theta = sum(temp(:,:,1:6),3);
     alpha = sum(temp(:,:,7:12),3);
     beta = sum(temp(:,:,13:19),3);
     gamma = sum(temp(:,:,20:29),3);
    flow_ad(i-2,:)= [reshape(theta',1,54*54) reshape(alpha',1,54*54) reshape(beta',1,54*54) reshape(gamma',1,54*54)];
     temp =mystand(EEG.EEG_microstates);
     for j=1:1:2000
         for k=1:1:20
         dis(k)=dist_E(temp(:,j),C(k,:)');
         end
         [~,state(j)]=min(dis);
     end
     feature_ad(i-2,:) = ts_analysis(state);
%     state =[state EEG.EEG_microstates];
end


for i=1:1:128
    for j=1:1:53
        X3(i,j)=fft_ad1(frequency_select(r(j),1),frequency_select(r(j),1),i);
    end
end

for i=1:1:30
[H1(i),~,~] =svmpredict(test_fft(i,end),test_fft(i,1:end-1), model_fft);
[H2(i),~,~] =svmpredict(test_cpl(i,end),test_cpl(i,1:end-1), model_cpl);
[H3(i),~,~] =svmpredict(test_flo(i,end),test_flo(i,1:end-1), model_flo);
[H4(i),~,~] =svmpredict(test_mic(i,end),test_mic(i,1:end-1), model_mic);
end

final=order1(1)*H1'+order1(2)*H2'+order1(3)*H3'+order1(4)*H4';
round(final);

%%%%%%%%%%%%连续模型拟合%%%%%%%
sheet = readtable('D:\SJN\0715mat\脑电认知整理.xlsx');
sheet = table2cell(sheet);

info1 = dir('D:\SJN\0715mat\process_new\test\ad');
for i=3:1:12
    load([info1(i).folder,'\',info1(i).name])
    name{i-2} = EEG.name{1,1}; 
%     fft_hc(:,:,i-2) =EEG.fft_result;
%     k = strcmp(sheet(:,2),name{i-2});
%     idx = find(k==1);
%     score(i-2) = sheet(k,12);
end

for i=1:1:124
    score1(i)=str2num(score{i});
end
r1 = unique(name);

c = setdiff(a,r1)% 删掉素组a中数组b的元素 如:

k = strcmp(sheet(:,2),name{2})
idx = find(k==1);

for i=1:1:54
   for j=1:1:1001
       aa = squeeze(fft_hc(i,j,:));
       rr(i,j)= corr(aa,score');
   end
end

r =reshape(rr',54054,1)';
[a b]=sort(abs(r),'descend');
for i=1:1:124
   aa = reshape(fft_hc(:,:,i)',54054,1)';
   train(i,:)=aa(:,b(1:100));   
end
%%%%%%%%%%%%%回归参数选择%%%%%%%%%%%%%
mse = 10^7;
for log2c = -10:0.5:3
    for log2g = -10:0.5:3
        % -v 交叉验证参数：在训练的时候需要，测试的时候不需要，否则出错
        options = ['-v 3 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g) , ' -s 3 -p 0.4 -t 3'];
        cv = libsvmtrain(label,train,options);
        if (cv < mse)
            mse = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%

options = ['-c ', num2str(2^bestc), ' -g ', num2str(2^bestg) , ' -s 3 -p 0.4 -n 0.1'];
model = libsvmtrain(label(1:100),train(1:100,:),options);
[predict_p,accuracy,dv] = svmpredict(label(101:end),train(101:end,:),model);
wu = sum(abs(label(101:end)-predict_p))/length(predict_p);

figure
plot(label(101:end),'o')
hold on
plot(predict_p,'.')