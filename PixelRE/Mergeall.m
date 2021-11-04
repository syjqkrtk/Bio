z = 100;
w = [5 7 10 12 14];
sp = [0 3 10 30 100];

for n = 1:5
    for m = 1:5
        if (w(n)==5) && (sp(m)==100)
        else
            title = sprintf('z%d_w%d_sp%d',z,w(n),sp(m));
            MergeImage();
            MergeImage2();
        end
    end
end