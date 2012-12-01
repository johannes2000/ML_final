function new_vect = feature_operation_joined_binary_with_interaction(Y1, Y2)
%Y1 and Y2 expected to be nx1, with values ranging from 1 to 10
%new_vect = is Nx(10+10+100)
%eg if Y1(1) = 4 and Y2(1) = 7, indices 4, 17, and 67 (20+30+6+1 = 57) new_vect
new_vect = sparse(numel(Y1),120);
for n = 1:numel(Y1)
    new_vect(n,Y1(n))=1;
    new_vect(n,Y2(n)+10)=1;
    new_vect(n,20+(Y1(n)-1)*10+Y2(n))=1;
end

%make new feature vector