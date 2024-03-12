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
import cmcrameri.cm as cmc
import os
# -

os.environ["SYMPY_USE_CACHE"] = "no"
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None

# +
res = 0.05
r_o = 1.0
r_int = 0.8
r_i = 0.5

free_slip_upper = True
free_slip_lower = True
# -

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize_Inner=res,
                                              cellSize_Internal=res*0.5,
                                              cellSize_Outer=res,
                                              filename="tmp_fixedstarsMesh.msh")


v_soln = uw.discretisation.MeshVariable("V0", meshball, 2, degree=2, varsymbol=r"{v_0}")
v_soln1 = uw.discretisation.MeshVariable("V1", meshball, 2, degree=2, varsymbol=r"{v_1}")
norm_v = uw.discretisation.MeshVariable("N", meshball, 2, degree=1, varsymbol=r"{\hat{n}}")
p_soln = uw.discretisation.MeshVariable("p", meshball, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("pc", meshball, 1, degree=1, continuous=True)

with meshball.access(norm_v):
    norm_v.data[:,0] = uw.function.evaluate(meshball.CoordinateSystem.unit_e_0[0], norm_v.coords)
    norm_v.data[:,1] = uw.function.evaluate(meshball.CoordinateSystem.unit_e_0[1], norm_v.coords)


def plot_vector(_mesh, _vector, _vector_name='', _cmap='', _clim='', _vmag='', _vfreq=''):
    # plot vector quantity from mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_vector_name] = vis.vector_fn_to_pv_points(pvmesh, _vector.sym)
    _vector_mag_name = _vector_name+'_mag'
    pvmesh.point_data[_vector_mag_name] = vis.scalar_fn_to_pv_points(pvmesh, 
                                                                     sympy.sqrt(_vector.sym.dot(_vector.sym)))
    
    print(pvmesh.point_data[_vector_mag_name].min(), pvmesh.point_data[_vector_mag_name].max())
    
    velocity_points = vis.meshVariable_to_pv_cloud(_vector)
    velocity_points.point_data[_vector_name] = vis.vector_fn_to_pv_points(velocity_points, _vector.sym)
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=_cmap, edge_color="Grey", scalars=_vector_mag_name, show_edges=False, use_transparency=False,
                opacity=0.7, clim=_clim )
    pl.add_arrows(velocity_points.points[::_vfreq], velocity_points.point_data[_vector_name][::_vfreq], mag=_vmag, color='k')

    pl.show(cpos="xy")


def plot_vector_nosym(_mesh, _vector, _vector_name='', _cmap='', _clim='', _vmag='', _vfreq=''):
    # plot vector quantity from mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_vector_name] = vis.vector_fn_to_pv_points(pvmesh, _vector)
    _vector_mag_name = _vector_name+'_mag'
    pvmesh.point_data[_vector_mag_name] = vis.scalar_fn_to_pv_points(pvmesh, 
                                                                     sympy.sqrt(_vector.dot(_vector)))
    
    print(pvmesh.point_data[_vector_mag_name].min(), pvmesh.point_data[_vector_mag_name].max())
    
    # velocity_points = vis.meshVariable_to_pv_cloud(_vector)
    # velocity_points.point_data[_vector_name] = vis.vector_fn_to_pv_points(velocity_points, _vector)
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=_cmap, edge_color="Grey", scalars=_vector_mag_name, show_edges=False, use_transparency=False,
                opacity=0.7, clim=_clim )
    pl.add_arrows(pvmesh.point_data[_vector_name][::_vfreq], pvmesh.point_data[_vector_name][::_vfreq], mag=_vmag, color='k')

    pl.show(cpos="xy")


# plotting analytical velocities
if uw.mpi.size == 1:
    plot_vector(meshball, norm_v, _vector_name='norm_v', _cmap=cmc.lapaz.resampled(11), _clim=[0., 1], _vmag=1e-1, _vfreq=1)

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

# Null space in velocity (constant v_theta) expressed in x,y coordinates
v_theta_fn_xy = r * meshball.CoordinateSystem.rRotN * sympy.Matrix((0,1))
# -


# plotting velocities
if uw.mpi.size == 1:
    plot_vector_nosym(meshball, r*unit_rvec, _vector_name='unit_rvec', _cmap=cmc.lapaz.resampled(11), _clim=[0., 1], _vmag=1e-1, _vfreq=1)

# plotting velocities
if uw.mpi.size == 1:
    plot_vector_nosym(meshball, v_theta_fn_xy.T, _vector_name='v_theta_fn_xy', _cmap=cmc.lapaz.resampled(11), _clim=[0., 1], _vmag=1e-1, _vfreq=1)

# +
#louis new code
# Null space evaluation

I0 = uw.maths.Integral(meshball, v_theta_fn_xy.dot(v_soln.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

print(norm/vnorm, vnorm)

with meshball.access(v_soln):
    dv = uw.function.evaluate(norm * v_theta_fn_xy, v_soln.coords) / vnorm
    v_soln.data[...] -= dv

# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.penalty = 0.0
stokes.saddle_preconditioner = sympy.simplify(1 / (stokes.constitutive_model.viscosity + stokes.penalty))

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

t_init = sympy.sin(5*th) * sympy.exp(-1000.0 * ((r - r_int) ** 2)) 


# +
stokes.bodyforce = sympy.Matrix([0,0])
Gamma = meshball.CoordinateSystem.unit_e_0

stokes.add_natural_bc(-t_init * unit_rvec, "Internal")

if free_slip_upper:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
else:
    stokes.add_essential_bc((0.0,0.0), "Upper")

if free_slip_lower:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
else:
    stokes.add_essential_bc((0.0,0.0), "Lower")

stokes.solve()
with meshball.access(v_soln1):
    v_soln1.data[...] = v_soln.data[...]


# -


# plotting velocities
if uw.mpi.size == 1:
    plot_vector(meshball, v_soln, _vector_name='v_soln', _cmap=cmc.lapaz.resampled(11), _clim=[0., .03], _vmag=5e0, _vfreq=10)

# +
# Null space evaluation

I0 = uw.maths.Integral(meshball, v_theta_fn_xy.dot(v_soln.sym))
norm = I0.evaluate()
I0.fn = v_soln.sym.dot(v_soln.sym)
vnorm = np.sqrt(I0.evaluate())


print(norm, vnorm)
# -

pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-3

# +
stokes._reset()

stokes.bodyforce = sympy.Matrix([0,0])
# Gamma = meshball.Gamma / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))
Gamma = norm_v.sym

stokes.add_natural_bc(-t_init * unit_rvec, "Internal")

if free_slip_upper:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
else:
    stokes.add_essential_bc((0.0,0.0), "Upper")

if free_slip_lower:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
else:
    stokes.add_essential_bc((0.0,0.0), "Lower")

stokes.solve()
# -

# plotting velocities
if uw.mpi.size == 1:
    plot_vector(meshball, v_soln, _vector_name='v_soln', _cmap=cmc.lapaz.resampled(11), _clim=[0., .03], _vmag=5e0, _vfreq=10)

# +
# Null space evaluation

I0 = uw.maths.Integral(meshball, v_theta_fn_xy.dot(v_soln.sym))
norm = I0.evaluate()
I0.fn = v_soln.sym.dot(v_soln.sym)
vnorm = np.sqrt(I0.evaluate())

print(norm, vnorm)

# -9.662093930530614e-09 0.024291704747453444
# -

# Pressure at mesh nodes
pressure_solver.solve()

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    velocity_points.point_data["V0"] = vis.vector_fn_to_pv_points(velocity_points, v_soln1.sym)
    velocity_points.point_data["dV"] = velocity_points.point_data["V"] - velocity_points.point_data["V0"]

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)
    pvmesh.point_data["V0"] = vis.vector_fn_to_pv_points(pvmesh, v_soln1.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["dV"] = pvmesh.point_data["V"] - pvmesh.point_data["V0"]
    pvmesh.point_data["Vmag"] = np.hypot(pvmesh.point_data["V"][:,0],pvmesh.point_data["V"][:,1])

    skip = 3
    points = np.zeros((meshball._centroids[::skip].shape[0], 3))
    points[:, 0] = meshball._centroids[::skip, 0]
    points[:, 1] = meshball._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="both", 
        integrator_type=2,
        surface_streamlines=True,
        initial_step_length=0.01,
        max_time=0.25,
        max_steps=500
    )
   

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        scalars="Vmag",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=True
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=1)
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V0"], mag=1, color="Black")
    pl.add_mesh(pvstream, opacity=0.3, show_scalar_bar=False)


    pl.show(cpos="xy")
# -

vsol_rms = np.sqrt(velocity_points.point_data["V"][:, 0] ** 2 + velocity_points.point_data["V"][:, 1] ** 2).mean()
vsol_rms

v_soln.sym-v_theta_fn_xy.T

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)

    
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["V_t"] = vis.vector_fn_to_pv_points(pvmesh, v_theta_fn_xy.T)
    pvmesh.point_data["dV1"] = pvmesh.point_data["V"] - pvmesh.point_data["V_t"]
    pvmesh.point_data["dV1mag"] = np.hypot(pvmesh.point_data["dV1"][:,0], pvmesh.point_data["dV1"][:,1])

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        scalars="dV1mag",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=True
    )

    pl.add_arrows(pvmesh.points, pvmesh.point_data["dV1"], mag=1e-1)


    pl.show(cpos="xy")
# -


