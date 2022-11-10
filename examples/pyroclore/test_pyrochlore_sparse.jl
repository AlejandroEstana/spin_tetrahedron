using LinearAlgebra
using KrylovKit
using SparseArrayKit
using JSON
using HDF5, JLD
using Random
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\sparse tensor\\spin_tetrahedron\\src\\reshape_tensor.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\sparse tensor\\spin_tetrahedron\\src\\conjugate_tensor.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\sparse tensor\\spin_tetrahedron\\src\\sparse_copy.jl")

cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\sparse tensor\\spin_tetrahedron\\examples\\pyroclore\\")
include("iPESS.jl")






Random.seed!(1234)


D=2;




is_sparse=true;
Bond_irrep="A";
Tetrahedral_irrep="E";

init_statenm=nothing;
init_noise=0;
json_state_dict, Bond_A_coe, Tetrahedral_E_coe=initial_state(is_sparse,Bond_irrep,Tetrahedral_irrep,D,init_statenm,init_noise)
Tetrahedral_E_coe[1]=1;
Tetrahedral_E_coe[2]=0;
json_state_dict=set_vector(json_state_dict, [1,1,0]);

A_set,E_set, S_label, Sz_label, virtual_particle=construct_tensor(D,is_sparse);
bond_tensor,tetrahedral_tensor=construct_su2_PG_IPESS(json_state_dict,A_set,E_set, S_label, Sz_label, virtual_particle);


Id=I(2);
sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
@tensor H12[:]:=sx[-1,-4]*sx[-2,-5]*Id[-3,-6]+sy[-1,-4]*sy[-2,-5]*Id[-3,-6]+sz[-1,-4]*sz[-2,-5]*Id[-3,-6];
@tensor H31[:]:=sx[-1,-4]*Id[-2,-5]*sx[-3,-6]+sy[-1,-4]*Id[-2,-5]*sy[-3,-6]+sz[-1,-4]*Id[-2,-5]*sz[-3,-6];
@tensor H23[:]:=Id[-1,-4]*sx[-2,-5]*sx[-3,-6]+Id[-1,-4]*sy[-2,-5]*sy[-3,-6]+Id[-1,-4]*sz[-2,-5]*sz[-3,-6];
@tensor H123chiral[:]:=sx[-1,-4]*sy[-2,-5]*sz[-3,-6]-sx[-1,-4]*sz[-2,-5]*sy[-3,-6]+sy[-1,-4]*sz[-2,-5]*sx[-3,-6]-sy[-1,-4]*sx[-2,-5]*sz[-3,-6]+sz[-1,-4]*sx[-2,-5]*sy[-3,-6]-sz[-1,-4]*sy[-2,-5]*sx[-3,-6];
H123chiral=sparse_copy(H123chiral);

@tensor tetrahedral_ov[:] := conjugate_tensor(tetrahedral_tensor)[1,2,3,4]*tetrahedral_tensor[1,2,3,4];
tetrahedral_ov=tetrahedral_ov[1];

@tensor chirality123[:] := conjugate_tensor(tetrahedral_tensor)[1,2,3,7]*H123chiral[4,5,6,1,2,3]*tetrahedral_tensor[4,5,6,7];
@tensor chirality243[:] := conjugate_tensor(tetrahedral_tensor)[7,1,3,2]*H123chiral[4,5,6,1,2,3]*tetrahedral_tensor[7,4,6,5];
@tensor chirality341[:] := conjugate_tensor(tetrahedral_tensor)[3,7,1,2]*H123chiral[4,5,6,1,2,3]*tetrahedral_tensor[6,7,4,5,];
@tensor chirality421[:] := conjugate_tensor(tetrahedral_tensor)[3,2,7,1]*H123chiral[4,5,6,1,2,3]*tetrahedral_tensor[6,5,7,4];

chirality123=chirality123/tetrahedral_ov;
chirality243=chirality243/tetrahedral_ov;
chirality341=chirality341/tetrahedral_ov;
chirality421=chirality421/tetrahedral_ov;

@tensor PEPS_part1[w,s,d,m,p1,p2,p3,p4] := bond_tensor[w,ww,p1]*bond_tensor[d,dd,p2]*bond_tensor[s,ss,p3]*bond_tensor[m,mm,p4]*tetrahedral_tensor[ww,dd,ss,mm];
PEPS_part1=reshape_tensor(PEPS_part1,5,8);
@tensor PEPS_tensor[w,s,e,n,u,d,p]:=PEPS_part1[w,s,d,m,p]*tetrahedral_tensor[u,e,n,m];
PEPS_tensor_dense=Array(PEPS_tensor);
println(varinfo(r"PEPS_tensor_dense"))
println(varinfo(r"PEPS_tensor"))

@tensor double_PEPS_part1[w1,w2,s1,s2,d1,d2,m1,m2]:=conjugate_tensor(PEPS_part1)[w1,s1,d1,m1,p]*PEPS_part1[w2,s2,d2,m2,p];
double_PEPS_part1=reshape_tensor(double_PEPS_part1,7,8);
double_PEPS_part1=reshape_tensor(double_PEPS_part1,5,6);
double_PEPS_part1=reshape_tensor(double_PEPS_part1,3,4);
double_PEPS_part1=reshape_tensor(double_PEPS_part1,1,2);
@tensor double_PEPS_part2[u1,u2,e1,e2,n1,n2,m1,m2]:=conjugate_tensor(tetrahedral_tensor)[u1,e1,n1,m1]*tetrahedral_tensor[u2,e2,n2,m2];
double_PEPS_part2=reshape_tensor(double_PEPS_part2,7,8);
double_PEPS_part2=reshape_tensor(double_PEPS_part2,5,6);
double_PEPS_part2=reshape_tensor(double_PEPS_part2,3,4);
double_PEPS_part2=reshape_tensor(double_PEPS_part2,1,2);
@tensor double_PEPS[w,s,e,n,u,d]:=double_PEPS_part1[w,s,d,m]*double_PEPS_part2[u,e,n,m];
double_PEPS_dense=Array(double_PEPS);
println(varinfo(r"double_PEPS_dense"))
println(varinfo(r"double_PEPS"))

# 2X2X2 cluster
@tensor layer_2D[u1,u2,u3,u4,d1,d2,d3,d4]:=double_PEPS[2,5,1,6,u1,d1]*double_PEPS[1,7,2,8,u2,d2]*double_PEPS[4,6,3,5,u3,d3]*double_PEPS[3,8,4,7,u4,d4];
layer_2D_dense=Array(layer_2D);
println(varinfo(r"layer_2D_dense"))
println(varinfo(r"layer_2D"))
sparse_rate=length(nonzero_keys(layer_2D))/prod(size(layer_2D))
println("Sparse rate: "*string(sparse_rate))

@tensor Norm_3D[:]:=layer_2D[5,6,7,8,1,2,3,4]*layer_2D[1,2,3,4,5,6,7,8];






