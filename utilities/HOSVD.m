function B = HOSVD(tensor,rank)
d = ndims(tensor);
B = tensor;
for i = 1:d
    mat = tensor_to_mat(tensor,i);
    [U,~,~] = svds(mat,rank(i));
    B = mul(B,i,U*U');
end
end
