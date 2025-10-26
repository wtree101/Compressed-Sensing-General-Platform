function tensor_mul = mul(tensor,index,V)
shape_new = size(tensor);
shape_new(index) = size(V,1);
tensor_mul = mat_to_tensor(V * tensor_to_mat(tensor,index),index,shape_new);
end
