
function sparse_copy(T,is_sparse=true)
    if is_sparse
        maxvalue=maximum(abs.(T));
        T_sparse=SparseArray{ComplexF64}(undef, size(T));
        for kk in keys(T)
            if abs(T[kk])/maxvalue >1e-12 
                T_sparse[kk] = T[kk]
            end
        end
        return T_sparse
    else
        return T
    end
end