% Correlation matrix 변수 초기화
clear CM;

% 데이터 행렬 및 이름 행렬 설정
DataMat = [cell2mat(divresult(1:divnum(1),1)); cell2mat(divresult(1:divnum(2),2)); cell2mat(divresult(1:divnum(3),3))];
NameMat = [divname(1:divnum(1),1); divname(1:divnum(2),2); divname(1:divnum(3),3)];

% Correlation 계산
for i = 1:size(DataMat,1)
    for j = 1:size(DataMat,1)
        xvar = mean(DataMat(i,:));
        yvar = mean(DataMat(j,:));
        sumxy = 0;
        sumx = 0;
        sumy = 0;
        for l = 1:size(DataMat,2)
            sumxy = sumxy + (DataMat(i,l)-xvar) * (DataMat(j,l)-yvar);
            sumx = sumx + (DataMat(i,l)-xvar)^2;
            sumy = sumy + (DataMat(j,l)-yvar)^2;
        end
        CM(i,j) = sumxy/sqrt(sumx*sumy);
    end
end

% Correlation matrix 값 기반 클러스터링 실행
n_m = 1:size(DataMat,1);

[a1 a2 a3 a4]  = clustering_algorithm160909(CM,n_m);

name2 = cell2mat(a4);
name3 = name(name2);
