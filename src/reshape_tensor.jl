
# Input:  tensor
#         first dimension
#         last dimension
# Output: tensor with the dimensions [first_dimension : last_dimension] contracted 
function reshape_tensor(in_tensor, first_dimension, last_dimension)
    in_tensor_keys = nonzero_keys(in_tensor)
    dimensions_to_reshape = range(first_dimension;stop=last_dimension)
    new_dimension_size = prod(size(in_tensor)[dimensions_to_reshape])
    new_dimensions = (size(in_tensor)[1:(first_dimension-1)]...,new_dimension_size, size(in_tensor)[(last_dimension + 1):end]...)
    out_tensor=SparseArray{ComplexF64}(undef, new_dimensions)
    for in_key in in_tensor_keys
        value = in_tensor[in_key]
        mapping = range(1; stop=new_dimension_size)
        mapping = reshape(mapping, size(in_tensor)[dimensions_to_reshape])
        tuple_key = Tuple(in_key)
        out_key = Int(mapping[tuple_key[first_dimension:last_dimension]...])
        out_tensor[tuple_key[1:(first_dimension - 1)]..., out_key, tuple_key[(last_dimension + 1):end]...] = value
    end
    out_tensor
end