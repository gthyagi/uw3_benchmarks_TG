# ## Annulus Benchmark: Isoviscous Incompressible Stokes
#
# #### [Benchmark ASPECT results](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/annulus/doc/annulus.html)
# #### [Benchmark paper](https://egusphere.copernicus.org/preprints/2023/egusphere-2023-2765/) 
#
# *Author: [Thyagarajulu Gollapalli](https://github.com/gthyagi)*

# ### Analytical solution

# This benchmark is based on a manufactured solution in which an analytical solution to the isoviscous incompressible Stokes equations is derived in an annulus geometry. The velocity and pressure fields are as follows:
#
# $$ v_{\theta}(r, \theta) = f(r) \cos(k\theta) $$
#
# $$ v_r(r, \theta) = g(r)k \sin(k\theta) $$
#
# $$ p(r, \theta) = kh(r) \sin(k\theta) + \rho_0g_r(R_2 - r) $$
#
# $$ \rho(r, \theta) = m(r)k \sin(k\theta) + \rho_0 $$
#
# with
#
# $$ f(r) = Ar + \frac{B}{r} $$
#
# $$ g(r) = \frac{A}{2}r + \frac{B}{r}\ln r + \frac{C}{r} $$
#
# $$ h(r) = \frac{2g(r) - f(r)}{r} $$
#
# $$ m(r) = g''(r) - \frac{g'(r)}{r} - \frac{g(r)}{r^2}(k^2 - 1) + \frac{f(r)}{r^2} + \frac{f'(r)}{r} $$
#
# $$ A = -C\frac{2(\ln R_1 - \ln R_2)}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
# $$ B = -C\frac{R_2^2 - R_1^2}{R_2^2 \ln R_1 - R_1^2 \ln R_2} $$
#
#
# The parameters $A$ and $B$ are chosen so that $ v_r(R_1, \theta) = v_r(R_2, \theta) = 0 $ for all $\theta \in [0, 2\pi]$, i.e. the velocity is tangential to both inner and outer surfaces. The gravity vector is radial inward and of unit length.
#
# The parameter $k$ controls the number of convection cells present in the domain
#
# In the present case, we set $ R_1 = 1, R_2 = 2$ and $C = -1 $.

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
# -

os.environ["SYMPY_USE_CACHE"] = "no"

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

# +
# radii
r_i = 1
r_o = 2

res = 1/32
# -

# visualize analytical solutions
plot_ana = True

# ### Analytical solution in sympy

# +
# analytical solution

r = sympy.symbols('r')
theta = sympy.Symbol('theta', real=True)

k=4
C=-1
A = -C*(2*(np.log(r_i) - np.log(r_o))/((r_o**2)*np.log(r_i) - (r_i**2)*np.log(r_o)))
B = -C*((r_o**2 - r_i**2)/((r_o**2)*np.log(r_i) - (r_i**2)*np.log(r_o)))
rho_0 = 0

f = sympy.Function('f')(r)
f = A*r + B/r

g = sympy.Function('g')(r)
g = ((A/2)*r) + ((B/r) * sympy.ln(r)) + (C/r)

h = sympy.Function('h')(r)
h = (2*g - f)/r

m = sympy.Function('m')(r)
m = g.diff(r, r) - (g.diff(r)/r) - (g/r**2)*(k**2 - 1) + (f/r**2) + (f.diff(r)/r)

v_r = g*k*sympy.sin(k*theta)
v_theta = f*sympy.cos(k*theta)
p = k*h*sympy.sin(k*theta) + rho_0*(r_o-r)
rho = m*k*sympy.sin(k*theta) + rho_0
v_x = v_r * sympy.cos(theta) - v_theta * sympy.sin(theta) 
v_y = v_r * sympy.sin(theta) + v_theta * sympy.cos(theta)

# v_r_ldy = lambdify((r, theta), v_r, modules=['numpy'])
# v_theta_ldy = lambdify((r, theta), v_theta, modules=['numpy'])
# p_ldy = lambdify((r, theta), p, modules=['numpy'])
# rho_ldy = lambdify((r, theta), rho, modules=['numpy'])
# -

# ### Create Mesh

# mesh
mesh = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=res, qdegree=2, degree=1)

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

# Analytical velocities
with mesh.access(v_ana, p_ana, rho_ana):
    v_ana_expr = mesh.CoordinateSystem.rRotN.T * sympy.Matrix([v_r.subs({r:r_uw, theta:th_uw}), 
                                                               v_theta.subs({r:r_uw, theta:th_uw})])
    v_ana.data[:,0] = uw.function.evalf(v_ana_expr[0], v_ana.coords)
    v_ana.data[:,1] = uw.function.evalf(v_ana_expr[1], v_ana.coords)

    p_ana.data[:,0] = uw.function.evalf(p.subs({r:r_uw, theta:th_uw}), p_ana.coords)

    rho_ana.data[:,0] = uw.function.evalf(rho.subs({r:r_uw, theta:th_uw}), rho_ana.coords)


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
    plot_vector(mesh, v_ana, _vector_name='v_ana', _cmap=cmc.lapaz.resampled(11), _clim=[0., 2.3], _vmag=1e-1, _vfreq=20)

# plotting analytical pressure
if uw.mpi.size == 1:
    plot_scalar(mesh, p_ana.sym, 'p_ana', _cmap=cmc.vik.resampled(41), _clim=[-8.5, 8.5])

# plotting analytical density
if uw.mpi.size == 1:
    plot_scalar(mesh, rho_ana.sym, 'rho', _cmap=cmc.roma.resampled(31), _clim=[-67.5, 67.5])

# +
# Create Stokes object
stokes = Stokes(mesh, velocityField=v_uw, pressureField=p_uw, solver_name="stokes")

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# gravity
gravity_fn = -1.0 * unit_rvec

# density
rho_uw = rho.subs({r:r_uw, theta:th_uw})

# bodyforce term
stokes.bodyforce = rho_uw*gravity_fn

# +
# boundary conditions

Gamma = mesh.Gamma
# Gamma = mesh.CoordinateSystem.unit_e_0
stokes.add_natural_bc(2.5e3 * Gamma.dot(v_uw.sym) *  Gamma, "Upper")
stokes.add_natural_bc(2.5e3 * Gamma.dot(v_uw.sym) *  Gamma, "Lower")


# # no slip
# stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Upper")
# stokes.add_essential_bc(sympy.Matrix([0., 0.]), "Lower")

# free slip
# Gamma = mesh.Gamma
# stokes.add_essential_bc(1e5 * Gamma.dot(v_uw.sym) *  Gamma, "Upper")
# stokes.add_essential_bc(1e5 * Gamma.dot(v_uw.sym) *  Gamma, "Lower")

# stokes.add_essential_bc(sympy.Matrix([v_ana_uw.sym[0], v_ana_uw.sym[1]]), "Upper")
# stokes.add_essential_bc(sympy.Matrix([v_ana_uw.sym[0], v_ana_uw.sym[1]]), "Lower")
# -

x, y = mesh.CoordinateSystem.N

Gamma.subs({x:x, y:y})

uw.function.evalf(Gamma, mesh.data)

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

stokes.solve(verbose=True)

# +
# # Null space in velocity (constant v_theta) expressed in x,y coordinates
# v_theta_fn_xy = r_uw * mesh.CoordinateSystem.rRotN * sympy.Matrix((0,1))

# +
# v_theta_fn_xy

# +
# I1 = uw.maths.Integral(mesh, v_uw.sym.dot(v_uw.sym))
# I1.evaluate()

# +
# # Null space evaluation

# I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v_uw.sym))
# norm = I0.evaluate()
# I0.fn = v_uw.sym.dot(v_uw.sym)
# vnorm = np.sqrt(I0.evaluate())

# print(norm, vnorm)
# -

# compute error
with mesh.access(v_uw, p_uw, v_err, p_err):

    v_ana_expr = mesh.CoordinateSystem.rRotN.T * sympy.Matrix([v_r.subs({r:r_uw, theta:th_uw}), 
                                                               v_theta.subs({r:r_uw, theta:th_uw})])
    v_err.data[:,0] = v_uw.data[:,0] - uw.function.evalf(v_ana_expr[0], v_err.coords)
    v_err.data[:,1] = v_uw.data[:,1] - uw.function.evalf(v_ana_expr[1], v_err.coords)

    p_err.data[:,0] = p_err.data[:,0] - uw.function.evalf(p.subs({r:r_uw, theta:th_uw}), p_err.coords)

# plotting velocities from uw
if uw.mpi.size == 1:
    plot_vector(mesh, v_uw, _vector_name='v_uw', _cmap=cmc.lapaz.resampled(11), _clim=[0., 2.3], _vfreq=20, _vmag=1e-1)

# plotting errror in velocities
if uw.mpi.size == 1:
    plot_vector(mesh, v_err, _vector_name='v_err', _cmap=cmc.lapaz.resampled(11), _clim=[0., 2.3], _vfreq=20, _vmag=1e-1)

# plotting pressure from uw
if uw.mpi.size == 1:
    plot_scalar(mesh, p_uw.sym, 'p_uw', _cmap=cmc.vik.resampled(41), _clim=[-8.5, 8.5])

# plotting error in uw
if uw.mpi.size == 1:
    plot_scalar(mesh, p_err.sym, 'p_err', _cmap=cmc.vik.resampled(41), _clim=[-8.5, 8.5])

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


