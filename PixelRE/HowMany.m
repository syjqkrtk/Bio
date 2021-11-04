function result = HowMany(Pattern)

result = zeros(59,59);
img = zeros(59,59);

for x = 1:59
    for y = 1:59
        temp = cell2mat(Pattern(x,y));
        result(x,y)=sum(sum(temp(:,:)));
    end
end

for i = 1:59
    for j = 1:59
        img(i,j) = 1-result(j,60-i)/max(max(result));
    end
end

imwrite(img, 'Data\59 MT genome Pattern.bmp');