using LinearAlgebra
using TensorOperations
using SparseArrayKit
using KrylovKit
include("reshape_tensor.jl")
include("conjugate_tensor.jl")

Nv=6;#options: 4,6,8
D=3;

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


A_fused = reshape_tensor(A,5,7);
println(varinfo(r"A_fused"))
#A_again = reshape_tensor(A_reshaped,5,(2,2,2));








function transfer_operator_fun_econ(x0,A,Nv,D)
    x=deepcopy(x0);

    #print(A)
    @tensor AA[:]:=conjugate_tensor(A)[-1,-3,-5,-7,1]*A[-2,-4,-6,-8,1];
    AA=reshape_tensor(AA,7,8);
    AA=reshape_tensor(AA,5,6);
    AA=reshape_tensor(AA,3,4);
    AA=reshape_tensor(AA,1,2);

    if Nv==4
        x=reshape_tensor(x,1,(D^2,D^2,D^2,D^2));
    elseif Nv==6
        x=reshape_tensor(x,1,(D^2,D^2,D^2,D^2,D^2,D^2));
    elseif Nv==8
        x=reshape_tensor(x,1,(D^2,D^2,D^2,D^2,D^2,D^2,D^2,D^2));
    end


    if Nv==4
        @tensor T_x[:]:=x[1,3,5,7]*AA[1,2,-1,8]*AA[3,4,-2,2]*AA[5,6,-3,4]*AA[7,8,-4,6];
    elseif Nv==6
        @tensor T_x[:]:=x[1,3,5,7,9,11]*AA[1,2,-1,12]*AA[3,4,-2,2]*AA[5,6,-3,4]*AA[7,8,-4,6]*AA[9,10,-5,8]*AA[11,12,-6,10];
    elseif Nv==8
        @tensor T_x[:]:=x[1,3,5,7,9,11,13,15]*AA[1,2,-1,16]*AA[3,4,-2,2]*AA[5,6,-3,4]*AA[7,8,-4,6]*AA[9,10,-5,8]*AA[11,12,-6,10]*AA[13,14,-7,12]*AA[15,16,-8,14];
    end

    T_x=reshape_tensor(T_x,1,length(size(T_x)));


    return T_x
end

#Create initial vector
@tensor AA_closed[:]:=conjugate_tensor(A_fused)[2,-1,-3,-5,1]*A_fused[2,-2,-4,-6,1];
AA_closed=reshape_tensor(AA_closed,5,6);
AA_closed=reshape_tensor(AA_closed,3,4);
AA_closed=reshape_tensor(AA_closed,1,2);
if Nv==4
    @tensor v0[:]:=AA_closed[1,-1,4]*AA_closed[2,-2,1]*AA_closed[3,-3,2]*AA_closed[4,-4,3];
elseif Nv==6
    @tensor v0[:]:=AA_closed[1,-1,6]*AA_closed[2,-2,1]*AA_closed[3,-3,2]*AA_closed[4,-4,3]*AA_closed[5,-5,4]*AA_closed[6,-6,5];
elseif Nv==8
    @tensor v0[:]:=AA_closed[1,-1,8]*AA_closed[2,-2,1]*AA_closed[3,-3,2]*AA_closed[4,-4,3]*AA_closed[5,-5,4]*AA_closed[6,-6,5]*AA_closed[7,-7,6]*AA_closed[8,-8,7];
end
v0=reshape_tensor(v0,1,length(size(v0)));



Tx_fun(x)=transfer_operator_fun_econ(x,A_fused,Nv,D);
@time euL,evL,info=eigsolve(Tx_fun, v0, 1,:LM, Arnoldi(krylovdim=10, tol=1e-14));

print(euL)



# def reshape_H(ev,D,Nv):
#     if Nv==4:
#         H=ev.reshape(D,D,D,D,D,D,D,D).transpose(0,2,4,6,1,3,5,7)
#     elif Nv==6:
#         H=ev.reshape(D,D,D,D,D,D,D,D,D,D,D,D).transpose(0,2,4,6,8,10,1,3,5,7,9,11)
#     elif Nv==8:
#         H=ev.reshape(D,D,D,D,D,D,D,D,D,D,D,D,D,D,D,D).transpose(0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15)
#     H=H.reshape(D**Nv,D**Nv)
#     H=H/np.trace(H)
#     return H


# #define matrix 
# fun=lambda x : transfer_operator_fun_econ(x,A,Nv,D)
# op = spla.LinearOperator(
#             matvec=fun,
#             dtype=A[0,0,0].dtype,
#             shape=(D**(Nv*2), D**(Nv*2))
# )
# eu,ev = spla.eigs(op, 2,  which="LM")
# print('Energy:')
# print(eu)
# euL1=eu[0];
# euL2=eu[1];
# HL1=reshape_H(ev[:,0],D,Nv)
# HL2=reshape_H(ev[:,1],D,Nv)



# #define matrix 
# fun=lambda x : transfer_operator_fun_econ(x,A.transpose(2,1,0,3,4),Nv,D)
# op = spla.LinearOperator(
#             matvec=fun,
#             dtype=A[0,0,0].dtype,
#             shape=(D**(Nv*2), D**(Nv*2))
# )
# eu,ev = spla.eigs(op, 2,  which="LM")
# print('Energy:')
# print(eu)
# euR1=eu[0];
# euR2=eu[1];
# HR1=reshape_H(ev[:,0],D,Nv)
# HR2=reshape_H(ev[:,1],D,Nv)





