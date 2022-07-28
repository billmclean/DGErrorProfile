# DGErrorProfile
Code for the paper "Error Profile for Discontinuous Galerkin Time 
Stepping of Parabolic PDE"

The tables and figures in the paper can be generated using the scripts
`table1.jl`, ..., `table4.jl` and `figure1.jl`, ..., `figure3.jl` in
the subdirectory `DGErrorProfile/scripts`.

The dependencies can be installed in the usual way by starting julia in
the `DGErrorProfile` subdirectory and doing

    (@v1.7) pkg> activate .
    (DGErrorProfile) pkg> instantiate

In addition, one unregistered dependency has to be handled manually:

    (DGErrorProfile) pkg> add https://github.com/billmclean/FractionalTimeDG.jl.git

You should see some meaningful parallel speedup by running julia with 
multiple threads for the 1D and 2D examples.
