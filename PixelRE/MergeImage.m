%% REminer 결과 그래프들을 전부 이어 큰 이미지로 만드는 프로그램
z = 100;
siz = z/10;

for ber = 1:siz
    clear result sizex sizey sizec img temp temp2 startx starty resultx resulty MTGenome i j k y2 fixx fixy Str;
    for num = 1:siz
        tic;
        Str = sprintf('Image_%s\\Screenshot_%d.png',title,num+(ber-1)*siz);
        img=double(imread(Str))/255; %0~255의 색상값을 가지는 이미지를 0~1로 바꿔주기 위함
        [sizey, sizex, sizec] = size(img);          %이미지 파일의 x,y,color 크기

        %% 이미지에서 그래프의 값 부분만 추출
        temp = rot90(img,3);

        startx = find((temp(:,sizey,1)==0),1);
        starty = find((temp(startx,:,1)==0),1);

        MTGenome = temp(startx+1:sizex,starty+1:sizey,:);
        
        MTGenome = 1 - MTGenome;
        [sizex, sizey, sizec] = size(MTGenome);         %실제 데이터 부분만의 사이즈를 다시 정의

        %% 이미지 이어붙이기
        if num == 1
            result = MTGenome;
        end

        result(1:800,800*(num-1)+1:800*num,:) = MTGenome(1:800,1:800,:);
        
        [resultx, resulty, resultc] = size(result);
        Str = sprintf('(%d/%d,%d/%d) image is now processing, %3.2f%% is completed', ber,siz, num,siz, ((ber-1)*siz+num)*100/(siz*siz));
        disp(Str);
        toc;
    end

    result = rot90(1-result);

    Str = sprintf('Image_%s\\Processed\\temp%d.bmp',title,ber);
    imwrite(result,Str);
end