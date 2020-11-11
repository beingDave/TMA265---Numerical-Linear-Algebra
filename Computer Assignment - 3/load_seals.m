function [x, y, class, hyp, cl_x1, cl_y1, cl_x2, cl_y2, seal_length, seal_weight] = load_seals
    seal_table = readtable('DatabaseGreySeal.xlsx');
    seal_year = seal_table.(3);
    seal_weight = seal_table.(10);
    seal_length = str2double(seal_table.(11));

    m = size(seal_weight,1);

    x=zeros(1,2*m);
    y=zeros(1,2*m);

    sch1=0;  sch2=0;

    class = zeros(1,m);
    hyp =   zeros(1,m);

    for i=1:1:m
        
        sch1 = sch1 +1;
        cl_y1(sch1)= seal_length(i);
        cl_x1(sch1) = seal_year(i);
        x(i) =  seal_year(i);

        sch2 = sch2 +1;
        cl_y2(sch2)= seal_weight(i);
        cl_x2(sch2) = seal_year(i);
        y(i) = seal_length(i);

        class(sch1) = 1;
        hyp(sch1)= 0;
    end

    i=0; sch3=0;

    for i=m+1:1:2*m
        sch3 = sch3 +1;
        y(i) = seal_weight(sch3);
        x(i) = seal_year(sch3);

        class(i) = 0;
        hyp(i)= 1;
    end

%     testSize = 0;
%     testSizeIndx = round(length(x)*(1-testSize));
%     xt = x(1,testSizeIndx+1:end); yt = y(1,testSizeIndx+1:end); ct = class(1,testSizeIndx+1:end);
%     x = x(1,1:testSizeIndx); y = y(1,1:testSizeIndx); class = class(1,1:testSizeIndx);
end