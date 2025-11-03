% Tensor matricization
function mat = tensor_to_mat(tensor,index)
    dim = ndims(tensor);
    x = size(tensor,index);
    y = 1;
    for i = 1:dim
        if (i~=index)
            y = y * size(tensor,i);
        end
    end
    
    tensor_s = permute(tensor,[index,1:index-1,index+1:dim]);
    mat = zeros(x,y);
    for i = 1:x
        mat(i,:) = reshape(tensor_s(i,:),1,[]);  %沿第i行展平
    end
end