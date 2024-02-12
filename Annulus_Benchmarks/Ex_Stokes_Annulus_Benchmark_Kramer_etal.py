# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# ### Case: Infinitely thin density anomaly at $r = r'$
# #### [Benchmark paper](https://gmd.copernicus.org/articles/14/1899/2021/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# +
import underworld3 as uw
from underworld3.systems import Stokes

import numpy as np
import sympy
from sympy import lambdify
import os
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import assess
# -

os.environ["SYMPY_USE_CACHE"] = "no"

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

# mesh options
res = 1/32
res_int_fac = 1/2
r_o = 2.0
r_int = 1.8
r_i = 1.0

# visualize analytical solutions
plot_ana = True

# boundary condition
freeslip = True
noslip = False

# +
# density perturbation
delta_fn = True
smooth = False

n = 2 # wave number
k = 2 # power (check the reference paper)
# -

# ### Analytical Solution

if plot_ana:
    if freeslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaFreeSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaFreeSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            # for smooth density distribution only single solution exists in the domain. But for sake of code optimization I am creating two solution here.
            soln_above = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothFreeSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=-1.0)
    elif noslip:
        if delta_fn:
            soln_above = assess.CylindricalStokesSolutionDeltaZeroSlip(n, +1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionDeltaZeroSlip(n, -1, Rp=r_o, Rm=r_i, rp=r_int, nu=1.0, g=-1.0)
        elif smooth:
            soln_above = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=-1.0)
            soln_below = assess.CylindricalStokesSolutionSmoothZeroSlip(n, k, Rp=r_o, Rm=r_i, nu=1.0, g=-1.0)

# ### Create Mesh

# mesh
mesh = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                          radiusInternal=r_int, 
                                          radiusInner=r_i, 
                                          cellSize_Inner=res,
                                          cellSize_Internal=res*res_int_fac,
                                          cellSize_Outer=res,)

if uw.mpi.size == 1:

    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, edge_color="Grey", show_edges=True, use_transparency=False, opacity=1.0, )

    pl.show(cpos="xy")

# +
# mesh variables
v_uw = uw.discretisation.MeshVariable(r"\mathbf{u}", mesh, 2, degree=2)
p_uw = uw.discretisation.MeshVariable(r"p", mesh, 1, degree=1)

if plot_ana:
    v_ana = uw.discretisation.MeshVariable(r"\mathbf{u_a}", mesh, 2, degree=2)
    p_ana = uw.discretisation.MeshVariable(r"p_a", mesh, 1, degree=1)
    rho_ana = uw.discretisation.MeshVariable(r"rho_a", mesh, 1, degree=1)
    
    v_err = uw.discretisation.MeshVariable(r"\mathbf{u_e}", mesh, 2, degree=2)
    p_err = uw.discretisation.MeshVariable(r"p_e", mesh, 1, degree=1)
# -

# Some useful coordinate stuff
unit_rvec = mesh.CoordinateSystem.unit_e_0
r_uw, th_uw = mesh.CoordinateSystem.xR

if plot_ana:
    with mesh.access(v_ana, p_ana, rho_ana):
        
        def get_ana_soln(_var, _r_int, _soln_above, _soln_below):
            # get analytical solution into mesh variables
            r = uw.function.evalf(r_uw, _var.coords)
            for i, coord in enumerate(_var.coords):
                if r[i]>_r_int:
                    _var.data[i] = _soln_above(coord)
                else:
                    _var.data[i] = _soln_below(coord)
                    
        # velocities
        get_ana_soln(v_ana, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)

        # pressure 
        get_ana_soln(p_ana, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
        
        # density
        get_ana_soln(rho_ana, r_int, soln_above.radial_stress_cartesian, soln_below.radial_stress_cartesian)
        

def plot_scalar(_mesh, _scalar, _scalar_name='', _cmap='', _clim='' ):
    # plot scalar quantity from mesh
    pvmesh = vis.mesh_to_pv_mesh(_mesh)
    pvmesh.point_data[_scalar_name] = vis.scalar_fn_to_pv_points(pvmesh, _scalar)

    print(pvmesh.point_data[_scalar_name].min(), pvmesh.point_data[_scalar_name].max())
    
    pl = pv.Plotter(window_size=(750, 750))
    pl.add_mesh(pvmesh, cmap=_cmap, edge_color="Grey", scalars=_scalar_name, show_edges=False, 
                use_transparency=False, opacity=1.0, clim=_clim)

    pl.show(cpos="xy")


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


# plotting analytical velocities
if uw.mpi.size == 1 and plot_ana:
    plot_vector(mesh, v_ana, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(11), _clim=[0., 0.05], _vmag=1e1, _vfreq=75)

# plotting analytical pressure
if uw.mpi.size == 1:
    plot_scalar(mesh, p_ana.sym, 'p_ana', _cmap=cmc.vik.resampled(41), _clim=[-0.65, 0.65])

# plotting analytical density
if uw.mpi.size == 1:
    plot_scalar(mesh, rho_ana.sym, 'Radial Stress', _cmap=cmc.roma.resampled(31), _clim=[-1, 1])

# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw, solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0
stokes.bodyforce = sympy.Matrix([0., 0.])

# +
# bc
rho = sympy.cos(n*th_uw) * sympy.exp(-1e5 * ((r_uw - r_int) ** 2)) 

if freeslip:
    Gamma = mesh.Gamma
    # Gamma = mesh.CoordinateSystem.unit_e_0
    stokes.add_natural_bc(2.5e3 * Gamma.dot(v_uw.sym) *  Gamma, "Upper")
    stokes.add_natural_bc(2.5e3 * Gamma.dot(v_uw.sym) *  Gamma, "Lower")
    stokes.add_natural_bc(-rho * unit_rvec, "Internal")
elif noslip:
    stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Upper")
    stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Lower")
# -

if uw.mpi.size == 1:
    plot_scalar(mesh, rho, 'rho', _cmap=cmc.roma.resampled(31), _clim=[-1, 1])

# +
# Stokes settings

stokes.tolerance = 1.0e-5
stokes.petsc_options["ksp_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

# stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# gasm is super-fast ... but mg seems to be bulletproof
# gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")
# -

stokes.solve()

# compute error
with mesh.access(v_uw, p_uw, v_err, p_err):

    def get_error(_var_err, _var_uw, _r_int, _soln_above, _soln_below):
        # get error in numerical solution
        r = uw.function.evalf(r_uw, _var_err.coords)
        for i, coord in enumerate(_var_err.coords):
            if r[i]>_r_int:
                _var_err.data[i] = _var_uw.data[i] - _soln_above(coord)
            else:
                _var_err.data[i] = _var_uw.data[i] - _soln_below(coord)
                
    # error in velocities
    get_error(v_err, v_uw, r_int, soln_above.velocity_cartesian, soln_below.velocity_cartesian)
    
    # error in pressure 
    get_error(p_err, p_uw, r_int, soln_above.pressure_cartesian, soln_below.pressure_cartesian)
    

# plotting velocities from uw
if uw.mpi.size == 1:
    plot_vector(mesh, v_uw, _vector_name='v_uw', _cmap=cmc.lapaz.resampled(11), _clim=[0., 0.05], _vfreq=75, _vmag=1e1)

# plotting errror in velocities
if uw.mpi.size == 1:
    plot_vector(mesh, v_err, _vector_name='v_err', _cmap=cmc.lapaz.resampled(11), _clim=[0., 0.005], _vfreq=75, _vmag=1e2)

# plotting pressure from uw
if uw.mpi.size == 1:
    plot_scalar(mesh, p_uw.sym, 'p_uw', _cmap=cmc.vik.resampled(41), _clim=[-0.65, 0.65])

# plotting error in uw
if uw.mpi.size == 1:
    plot_scalar(mesh, p_err.sym, 'p_err', _cmap=cmc.vik.resampled(41), _clim=[-0.065, 0.065])

# computing L2 norm
with mesh.access(v_err, p_err, p_ana, v_ana):    
    v_err_I = uw.maths.Integral(mesh, v_err.sym.dot(v_err.sym))
    v_ana_I = uw.maths.Integral(mesh, v_ana.sym.dot(v_ana.sym))
    v_err_l2 = np.sqrt(v_err_I.evaluate())/np.sqrt(v_ana_I.evaluate())
    print('Relative error in velocity in the L2 norm: ', v_err_l2)

    p_err_I = uw.maths.Integral(mesh, p_err.sym.dot(p_err.sym))
    p_ana_I = uw.maths.Integral(mesh, p_ana.sym.dot(p_ana.sym))
    p_err_l2 = np.sqrt(p_err_I.evaluate())/np.sqrt(p_ana_I.evaluate())
    print('Relative error in pressure in the L2 norm: ', p_err_l2)

# +
# # Gamma = mesh.Gamma
# # for penalty value = 2.5e3
# Relative error in velocity in the L2 norm:  0.025879215028494718
# Relative error in pressure in the L2 norm:  0.12054560357965426

# # for penalty value = 2.5e6
# Relative error in velocity in the L2 norm:  0.7854162259503036
# Relative error in pressure in the L2 norm:  0.3699040718486331

# # Gamma = mesh.CoordinateSystem.unit_e_0
# # for penalty value = 2.5e3
# Relative error in velocity in the L2 norm:  0.012508614854054356
# Relative error in pressure in the L2 norm:  0.12032615004615484

# # for penalty value = 2.5e6
# Relative error in velocity in the L2 norm:  0.01441492658727351
# Relative error in pressure in the L2 norm:  0.12035923615727437
# -


