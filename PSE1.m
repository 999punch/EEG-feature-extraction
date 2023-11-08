function [PSE] = PSE1(data,srate)
%   计算信号的功率谱熵
%   data为输入信号，srate为采样频率，PSE为功率谱熵
PSE=[0 0 0 0 0 0];
c=20;   %此处c的倒数代表功率谱密度图的分辨率
[pxx,f] = pwelch(data,hanning(2*srate),0,c*srate,srate);
% 这里hanning(2*srate)表示汉宁窗的长度为2*srate个采样点，即2秒；
% 0表示各段之间重叠的点数为0；
% c*srate表示功率谱密度图的分辨率为1/c，即每(1/c)Hz计算一次；
% srate表示采样频率；

PSE(1) = sum(pxx(0.5*c+1:4*c+1,1).*(-log2(pxx(0.5*c+1:4*c+1,1))));
PSE(2) = sum(pxx(4*c+1:8*c+1,1).*(-log2(pxx(4*c+1:8*c+1,1))));
PSE(3) = sum(pxx(8*c+1:13*c+1,1).*(-log2(pxx(8*c+1:13*c+1,1))));
PSE(4) = sum(pxx(13*c+1:30*c+1,1).*(-log2(pxx(13*c+1:30*c+1,1))));
PSE(5) = sum(pxx(30*c+1:100*c+1,1).*(-log2(pxx(30*c+1:100*c+1,1))));
PSE(6) = sum(pxx(0*c+1:100*c+1,1).*(-log2(pxx(0*c+1:100*c+1,1))));

psd = 10*log10(pxx);
plot(f(1:100*c),psd(1:100*c,1),'r',f(1:100*c),psd(1:100*c,2),'g',f(1:100*c),psd(1:100*c,3),'b');    %此段为功率谱密度图演示
grid;
xlabel('频率,Hz')
ylabel('功率谱密度')

end