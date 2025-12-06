import Pkg
Pkg.activate(joinpath(@__DIR__, "environment"))

include(joinpath(@__DIR__, "utils", "ML1Utils.jl"))
include(joinpath(@__DIR__, "utils", "RoiSubjectDisjoint.jl"))
include(joinpath(@__DIR__, "utils", "BurakPCA.jl"))
include(joinpath(@__DIR__, "utils", "ErikWISMAutoencode.jl"))

using .ML1Utils
using .RoiSubjectDisjoint
using .BurakPCA
using .ErikWISMAutoencode


function main()
    res_pca     = run_pca_approach(; seed = 1234)
    res_wism    = run_wism_autoencode_approach(; seed=1234)
    res_subject = run_subject_disjoint_approach(; seed=1234)
end

main()