function [x, y, class, xt, yt, ct, cl_x1, cl_y1, cl_x2, cl_y2] = load_iris
    data = csvread("iris.csv");

    Xiris = data(:, 1)'; Yiris = data(:, 2)'; Ciris = data(:, 3)';
    x = Xiris(1, :); y = Yiris(1, :); class = Ciris(1, :);

    sortIdx = randperm(length(x));
    x = x(sortIdx);
    y = y(sortIdx);
    class = class(sortIdx);

    testSizeIndx = 70;
    xt = x(1,testSizeIndx+1:100); yt = y(1,testSizeIndx+1:100); ct = class(1,testSizeIndx+1:100);
    x = x(1,1:testSizeIndx); y = y(1,1:testSizeIndx); class = class(1,1:testSizeIndx);

    cl_x1 = x(class==1); cl_y1 = y(class==1);
    cl_x2 = x(class==0); cl_y2 = y(class==0);

    clt_x1 = xt(ct==1); clt_y1 = yt(ct==1);
    clt_x2 = xt(ct==0); clt_y2 = yt(ct==0);
end