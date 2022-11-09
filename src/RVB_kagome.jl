using LinearAlgebra
using TensorOperations
using SparseArrayKit
include("reshape_tensor.jl")

b=SparseArray{ComplexF64}(undef, (3,3,2));
b[1,2,1]=1/2;
b[2,1,1]=1/2;
b[3,1,2]=1/2;
b[1,3,2]=1/2;
t=SparseArray{ComplexF64}(undef, (3,3,3));
t[1,1,1]=1;
t[3,2,1]=1/sqrt(6);
t[2,3,1]=-1/sqrt(6);
t[3,1,2]=-1/sqrt(6);
t[1,3,2]=1/sqrt(6);
t[2,1,3]=1/sqrt(6);
t[1,2,3]=-1/sqrt(6);


@tensor A[a,b,c,d,e,f,g]:=b[a,i,e]*b[l,k,f]*b[d,j,g]*t[i,k,j]*t[l,b,c];
println(varinfo(r"A"))
A_dense=Array(A);
println(varinfo(r"A_dense"))

Unitary_phy_dense=Matrix(I, 2^3, 2^3);
Unitary_phy_dense=reshape(Unitary_phy_dense,(2,2,2,2^3));
Unitary_phy=SparseArray{ComplexF64}(undef, size(Unitary_phy_dense));
for kk in keys(Unitary_phy_dense)
    if abs(Unitary_phy_dense[kk]) >0 
        Unitary_phy[kk] = Unitary_phy_dense[kk]
    end
end
println(varinfo(r"Unitary_phy_dense"))
println(varinfo(r"Unitary_phy"))

@tensor A_fused[a,b,c,d,h]:=A[a,b,c,d,e,f,g]*Unitary_phy[e,f,g,h]

# (Tensor_to_be_reshaped, first_dimension, second_dimension)
A_reshaped = reshape_tensor(A,5,7)



# TODO Do the reshape function from this code. From many indexes to One
A_keys = nonzero_keys(A)
A_reshaped=SparseArray{ComplexF64}(undef, (3,3,3,3,8));
size(A)

#TODO esto es para el (from one to many)
key_dict = Matrix{Int}(undef, (8,3))
for key in A_keys
    # println(key)
    value = A[key]
    
    mapping = range(1, stop=8, length=8)
    mapping = reshape(mapping, 2,2,2)
    new_key = Int(mapping[key[5],key[6],key[7]])

    #TODO esto es para el (from one to many)
    key_dict[new_key, 1:3] = [key[5],key[6],key[7]]
    # println(value, " ", new_key)
    A_reshaped[key[1],key[2],key[3],key[4], new_key] = value
    # print(key[1])
end
