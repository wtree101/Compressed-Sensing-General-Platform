function tensor = mat_to_tensor(mat,index,shape)
    tensor = zeros(shape);
    shape_except_index = shape;
    shape_except_index(index) = [];
    permute_order = [index, 1:index-1, index+1:size(shape,2)];
    tensor = permute(tensor, permute_order);
    for i = 1:shape(index)
        tensor_index = repmat({':'}, 1, ndims(tensor));
        tensor_index{1} = i;  
        if numel(shape_except_index) > 1
            tensor(tensor_index{:}) = reshape(mat(i, :), shape_except_index);
        else
            tensor(tensor_index{:}) = mat(i, :);
        end
    end
    tensor = ipermute(tensor, permute_order);
end
