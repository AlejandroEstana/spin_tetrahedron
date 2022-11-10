using MAT
using TensorKit
function construct_tensor(D,is_sparse)
    #D=3
    filenm="bond_tensors_D_"*string(D)*".mat"
    vars = matread(filenm)
    A_set=vars["A_set"][1,:]
    A_set_occu=Vector(undef,length(A_set))
    S_label=vars["S_label"][1,:]
    Sz_label=vars["Sz_label"][1,:]
    virtual_particle=vars["virtual_particle"][1,:]
    #typeof(A_set[1]["tensor"])

    filenm="Tetrahedral_tensors_D_"*string(D)*".mat"
    vars = matread(filenm)
    E_set=vars["E_set"][1,:]
    E_set_occu=Vector(undef,length(E_set))
    Va=[]
    Vb=[]

    for cm=1:length(A_set)
        A_set_occu[cm]=A_set[cm]["sectors"]
        A_set[cm]=sparse_copy(A_set[cm]["tensor"],is_sparse)
    end



    for cm=1:length(E_set)
        E_set_occu[cm]=E_set[cm]["sectors"]
        E_set[cm]=sparse_copy(E_set[cm]["tensor"],is_sparse)
    end

    
    return A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb;
end;