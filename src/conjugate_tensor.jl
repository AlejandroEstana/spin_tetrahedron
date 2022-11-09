
# Input:  tensor
#         first dimension
#         last dimension
# Output: tensor with the dimensions [first_dimension : last_dimension] contracted 
function conjugate_tensor(in_tensor)
    in_tensor_keys = nonzero_keys(in_tensor)
    out_tensor=SparseArray{ComplexF64}(undef, size(in_tensor))
    for in_key in in_tensor_keys
        out_tensor[in_key] = conj(in_tensor[in_key])
    end
    out_tensor
end

