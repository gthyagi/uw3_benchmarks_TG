# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Cylindrical Stokes 
#
# Mesh with embedded internal surface
#
# This allows us to introduce an internal force integral

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy
import os
import cmcrameri.cm as cmc
# -

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

# +
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

os.environ["SYMPY_USE_CACHE"] = "no"

# +
# mesh options
res = 0.05
r_o = 1.0
r_int = 0.8
r_i = 0.5

free_slip_upper = True
# -

meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize_Inner=res,
                                              cellSize_Internal=res*0.5,
                                              cellSize_Outer=res,)


v_soln = uw.discretisation.MeshVariable(r"\mathbf{u}", meshball, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(r"p", meshball, 1, degree=1, continuous=False)
p_cont = uw.discretisation.MeshVariable(r"p", meshball, 1, degree=1, continuous=True)

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis
    pv.global_theme.jupyter_backend = "none"
    pvmesh = vis.mesh_to_pv_mesh(meshball)
   
    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh,
                edge_color="Grey",
                show_edges=True,
                use_transparency=False,
                opacity=1.0,
                )

    pl.show(cpos="xy")

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

radius_fn = meshball.CoordinateSystem.xR[0]
unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = 1  # radius_fn / r_o

# Some useful coordinate stuff

x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.xR

Rayleigh = 1.0e5
# -


meshball.dm.view()

# +
# sympy.exp(0)

# +
# Create Stokes object

stokes = Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes"
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

t_init = sympy.sin(5*th) * sympy.exp(-1000000.0 * ((r - r_int) ** 2)) 

Gamma = meshball.Gamma
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
stokes.add_natural_bc(-t_init * unit_rvec, "Internal")

stokes.bodyforce = sympy.Matrix([0,0])
# -


if uw.mpi.size == 1:
    pvmesh = vis.mesh_to_pv_mesh(meshball)
    pvmesh.point_data["rho"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)

    print(pvmesh.point_data["rho"].min(), pvmesh.point_data["rho"].max())
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=cmc.roma.resampled(31), edge_color="Grey",
                scalars="rho", show_edges=False, use_transparency=False,
                opacity=1.0, clim=[-1, 1] )

    pl.show(cpos="xy")

pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)
stokes.solve()

# Pressure at mesh nodes
pressure_solver.solve()

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis
    pv.global_theme.jupyter_backend = "none"
    pvmesh = vis.mesh_to_pv_mesh(meshball)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)
   
    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh,
                cmap="coolwarm",
                edge_color="Grey",
                scalars="T",
                show_edges=False,
                use_transparency=False,
                opacity=1.0,
                clim=[-1, 1] #[-0.5, 0.5]
                )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=5)


    pl.show(cpos="xy")
# -

if uw.mpi.size == 1:
    vsol_rms = np.sqrt(velocity_points.point_data["V"][:, 0] ** 2 + velocity_points.point_data["V"][:, 1] ** 2).mean()
    print(vsol_rms)
else:
    with meshball.access():
        vsol_rms = np.sqrt(v_soln.data[:, 0] ** 2 + v_soln.data[:, 1] ** 2).mean()
        print(vsol_rms)

with meshball.access():
    print(v_soln.data)
    print(p_soln.data)




