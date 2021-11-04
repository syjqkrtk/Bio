a = result2.VarName1;
data = zeros(50,50);
count = 0;

for i = 1:49
    data(i,i) = 1;
    for j = i+1:50
        count = count + 1;
        data(i,j) = a(count);
        data(j,i) = a(count);
    end
end
data(50,50) = 1;

data2 = zeros(48,48);

data2(1:22,1:22) = data(1:22,1:22);
data2(23:48,1:22) = data(24:49,1:22);
data2(1:22,23:48) = data(1:22,24:49);
data2(23:48,23:48) = data(24:49,24:49);

result = kmedoids(1-data2,2);