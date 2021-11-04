% 파라미터 범위 및 결과값 변수 설정
z = 100;
w = [7 8 10 12 14];
sp = [0 3 10 30 100];
pix = zeros(5,5);
res = zeros(5,5);

% 파라미터별 파일 명 (title) 설정 및 클러스터링 코드 실행
for n = 1:size(w,2)
    for m = 1:size(sp,2)
        title = sprintf('z%d_w%d_sp%d',z,w(n),sp(m));
        REMinerPattern
        REMinerProcess
        function_test160909
        RandMeasure
        pix(n,m) = pixel;
        res(n,m) = RI;
    end
end