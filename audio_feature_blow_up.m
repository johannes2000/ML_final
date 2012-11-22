

xd_aud = xd_aud_orig;

%% just some feature blow up
bam = horzcat(xd_aud, xd_aud.^2, sqrt(xd_aud), 2.^(xd_aud)); 

%do transforms on all feature dimensions to make more normal
%hist(xd_aud(:,10))

%% Self multiply 
selfmult = bam; %the thing we would like to selfmultiply
k = size(selfmult,2); %number of features
output = selfmult;

for i = 1:k
    for j = i:k
        output = horzcat(output, selfmult(:,i).*selfmult(:,j)); %including every feature with itself
    end
end

%% scaling
data = output;
output_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

