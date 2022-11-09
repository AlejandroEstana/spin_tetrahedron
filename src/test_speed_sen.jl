using TensorOperations
using SparseArrayKit





sz=(100,100,100);
T1=SparseArray{ComplexF64}(undef, sz);
T1[1,1,1]=1;

println(varinfo(r"T1"))

T1_dense=Array(T1);
println(varinfo(r"T1_dense"))


@time @tensor T2[c,d]:=T1[a,b,c]*T1[a,b,d];
@time @tensor T2_dense[c,d]:=T1_dense[a,b,c]*T1_dense[a,b,d];

println(varinfo(r"T2"))
println(varinfo(r"T2_dense"))

@time @tensor T3[b,c,d,e]:=T1[a,b,c]*T1[a,d,e];
@time @tensor T3_dense[b,c,d,e]:=T1_dense[a,b,c]*T1_dense[a,d,e];
println(varinfo(r"T3"))
println(varinfo(r"T3_dense"))

@time TT1=reshape(T1,(sz[1],sz[2]*sz[3]));
#@time @tensor TT3[b,c]:=TT1[a,b]*TT1[a,c];
@time @tensor TT3[b,c]:=TT1[b,a]*TT1[c,a];
println(varinfo(r"TT1"))
