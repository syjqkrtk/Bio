%% Rand index
a5 = zeros(1,size(a1,1));
Real = [ones(1,divnum(1)) 2*ones(1,divnum(2)) 3*ones(1,divnum(3))];
TP = 0;
FP = 0;
FN = 0;
TN = 0;

for i = 1:a2
    for j = 1:size(a4{i},2)
        a5(a4{i}(j)) = i;
    end
end

%% Pair ºñ±³
for i = 1:54-1
    for j = i+1:54
        TP = TP + ((a5(i)==a5(j))&&(Real(i)==Real(j)));
        FP = FP + ((a5(i)==a5(j))&&(Real(i)~=Real(j)));
        FN = FN + ((a5(i)~=a5(j))&&(Real(i)==Real(j)));
        TN = TN + ((a5(i)~=a5(j))&&(Real(i)~=Real(j)));
    end
end

RI = (TP+TN)/(TP+FP+FN+TN);
pixel = sum(sum(cell2mat(Pattern)));