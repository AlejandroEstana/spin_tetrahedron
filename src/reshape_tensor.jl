
# Input:  tensor
#         first dimension
#         last dimension
# Output: tensor with the dimensions [first_dimension : last_dimension] contracted 
function reshape_tensor(in_tensor, first_dimension::Int, last_dimension::Int)
    in_tensor_keys = nonzero_keys(in_tensor)
    dimensions_to_reshape = range(first_dimension;stop=last_dimension)
    new_dimension_size = prod(size(in_tensor)[dimensions_to_reshape])
    new_dimensions = (size(in_tensor)[1:(first_dimension-1)]...,new_dimension_size, size(in_tensor)[(last_dimension + 1):end]...)
    out_tensor=SparseArray{ComplexF64}(undef, new_dimensions)
    mapping = range(1; stop=new_dimension_size)
    mapping = reshape(mapping, size(in_tensor)[dimensions_to_reshape])
    for in_key in in_tensor_keys
        value = in_tensor[in_key]
        tuple_key = Tuple(in_key)
        out_key = Int(mapping[tuple_key[first_dimension:last_dimension]...])
        out_tensor[tuple_key[1:(first_dimension - 1)]..., out_key, tuple_key[(last_dimension + 1):end]...] = value
    end
    out_tensor
end

# Input:  tensor
#         dimension to be expanded
#         new_dimensions
# Output: tensor with the dimensions [first_dimension : last_dimension] contracted 
function reshape_tensor(in_tensor, dimension::Int, new_dimensions::Tuple)
    in_tensor_keys = nonzero_keys(in_tensor)
    total_dimensions = (size(in_tensor)[1:(dimension-1)]...,new_dimensions..., size(in_tensor)[(dimension + 1):end]...)
    mapping = range(1; stop=size(in_tensor)[dimension])
    mapping = reshape(mapping, new_dimensions)
    invers_mapping = Dict()
    for indexes in keys(mapping)
        invers_mapping[mapping[indexes]] = indexes
    end
    out_tensor=SparseArray{ComplexF64}(undef, total_dimensions)
    for in_key in in_tensor_keys
        value = in_tensor[in_key]
        key_value = in_key[dimension]
        new_indexes = invers_mapping[key_value]
        tuple_new_indexes = Tuple(new_indexes)
        tuple_key = Tuple(in_key)
        out_tensor[tuple_key[1:(dimension - 1)]..., tuple_new_indexes..., tuple_key[(dimension + 1):end]...] = value
    end
    out_tensor
end