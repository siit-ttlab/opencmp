########################################################################################################################
# Copyright 2021 the authors (see AUTHORS file for full list).                                                         #
#                                                                                                                      #
# This file is part of OpenCMP.                                                                                        #
#                                                                                                                      #
# OpenCMP is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public  #
# License as published by the Free Software Foundation, either version 2.1 of the License, or (at your option) any     #
# later version.                                                                                                       #
#                                                                                                                      #
# OpenCMP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied        #
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more  #
# details.                                                                                                             #
#                                                                                                                      #
# You should have received a copy of the GNU Lesser General Public License along with OpenCMP. If not, see             #
# <https://www.gnu.org/licenses/>.                                                                                     #
########################################################################################################################

"""
Two-fluid model with a phase energy equation for each phase.

For each phase k (c: continuous/liquid, d: dispersed/gas) a temperature T_k is transported, written in
non-conservative (temperature) form and divided by the constant rho_k*cp_k:

    A_k (dT_k/dt + u_k . grad T_k) = div(A_k kappa_k grad T_k) + Q_ik/(rho_k cp_k) + S_k

    kappa_k = k_k/(rho_k cp_k)
    Q_ic    = h_i a_i (T_d - T_c),  Q_id = -Q_ic          (interphase sensible heat exchange)
    S_d     = mdot (T_sat - T_d)                         (gas created by the gas sources enters at T_sat)

This is the conservative form d(A_k T_k)/dt + div(A_k u_k T_k) minus T_k times the phase continuity equation. The
continuity part is discretized with the same upwind DG operator as the advection, so a uniform temperature stays
uniform independently of how TwoFluidModel discretizes alpha_c and the mixture continuity (a purely conservative
form turns any mismatch between those discretizations into spurious heating/cooling). Energy is then conserved up
to the discretization error of the phase continuity.

The phase fractions are lagged (old time level in the storage term, Picard iterate elsewhere). A small
regularization epsilon is added to the phase fraction in the storage and diffusion terms so that T_k stays defined
where phase k is absent.

Both temperatures use L2 (DG) spaces: upwind advective fluxes, SIPG diffusion.

Boundary conditions:
    - Walls (u_k.n = 0) are adiabatic unless given a heat flux in [NEUMANN] (W/m^2 into the domain, applied to
      the equation of the phase it is given for).
    - Open (ZERO_STRESS) boundaries of u_k: outflow advects T_k out; backflow enters as pure liquid at
      T_backflow.

Parameters are read from the [ENERGY] section of model_dir/model_config:
    cp_c, cp_d           specific heats [J/kg/K]
    k_c, k_d             thermal conductivities [W/m/K]
    T_sat                saturation temperature [K]
    T_backflow           temperature of liquid entering through open boundaries [K], default T_sat
    epsilon              phase fraction regularization, default 1e-3
    interphase_heat_transfer = RanzMarshall | none, default RanzMarshall

Optional [GAS_FREE] section: regularization of the gas velocity where there is (almost) no gas. Where alpha_d is
tiny the gas momentum equation barely constrains u_d, and in a gas-free cell u_d has no meaning; letting the
equations determine it there makes the Picard iteration cycle in flat wall cells and stall the run (see
docs/stall_low_heat_flux.md of the TFM-automata coupling). With the section, the gas momentum equation (per unit gas
mass) gets a pull of u_d towards the drift velocity u_c + U_r e_up (e_up against gravity),
    K_d (u_c + U_r e_up - u_d),   K_d = w(alpha_d) (1 + m) / tau,   m = C_VM rho_c/rho_d (0 without virtual mass),
    w(alpha_d) = (1 - min(alpha_d/alpha_min, 1))^2   (1 without gas, 0 from alpha_min on),
so that without gas u_d relaxes to the drift velocity over tau, the gas inertia including its added mass. The
liquid gets the equal and opposite reaction, K_c = -(alpha_d rho_d)/(alpha_c rho_c) K_d, as for drag and virtual
mass. alpha_d is taken from the Picard iterate like the other interphase terms. Without the section nothing changes.
    alpha_min            gas fraction from which the pull is off, default 1e-3
    relaxation_time      tau [s], default 1e-4
    drift_velocity       U_r: terminal (the terminal rise velocity of a bubble of diameter dp with the model's drag
                         closure) or a number [m/s], default terminal

TODO: physics to consider adding
    - Boussinesq buoyancy in the liquid momentum equation, rho_c beta (T_c - T_ref) g, for natural convection.
    - Interphase mass transfer: condensation of vapor in subcooled liquid (and evaporation in superheated liquid),
      with the latent heat going to the liquid and a gas volume sink/source in alpha_c and the mixture continuity.
    - Latent heat of the gas generated by the gas sources, as an alternative to removing it from the wall flux.
    - Temperature-dependent properties (rho, cp, k, nu, sigma), in particular for subcooled conditions.
    - Viscous dissipation and pressure work (negligible for pool boiling).
    - Turbulent / bubble-induced heat diffusivity.
    - Momentum of the gas created by the gas sources in the dispersed-phase momentum equation.
"""

import logging
import math
from typing import Dict, List, Optional, Union

import ngsolve as ngs
from ngsolve import GridFunction, FESpace, Parameter
from ngsolve.comp import ProxyFunction

from .tfm import TwoFluidModel
from ..helpers.dg import avg, jump, weighted_grad_avg
from ..helpers.math import Max, Min


class TwoFluidModelEnergy(TwoFluidModel):
    """
    TwoFluidModel with liquid and gas temperature equations, both in L2 spaces. The temperatures are the
    model components t_c and t_d (lower case, like the other variables, since config option names are
    lower-cased).
    """

    ENERGY_COMPONENTS = ('t_c', 't_d')
    GAS_FREE_OPTIONS = {'alpha_min', 'relaxation_time', 'drift_velocity'}
    INTERPHASE_HEAT_TRANSFER_MODELS = ('RanzMarshall', 'none')
    BC_VARIABLES = dict(TwoFluidModel.BC_VARIABLES, neumann=('t_c', 't_d'))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _define_model_components(self) -> Dict[str, Optional[int]]:
        components = super()._define_model_components()
        n = len(components)
        components.update({'t_c': n, 't_d': n + 1})
        return components

    def _define_model_local_error_components(self) -> Dict[str, bool]:
        return dict(super()._define_model_local_error_components(), t_c=True, t_d=True)

    def _define_time_derivative_components(self) -> List[Dict[str, bool]]:
        # The energy storage terms are phase-fraction weighted, so they are added in time_derivative_terms
        # instead of the default T*v terms.
        return [dict(forms, t_c=False, t_d=False) for forms in super()._define_time_derivative_components()]

    def _define_bc_types(self) -> List[str]:
        return super()._define_bc_types() + ['neumann']

    def _construct_fes(self) -> FESpace:
        for name in self.ENERGY_COMPONENTS:
            if self.element[name] != 'L2':
                raise ValueError("TwoFluidModelEnergy requires an L2 element for '{}', got '{}'."
                                 .format(name, self.element[name]))

        spaces = list(super()._construct_fes().components)
        scalar_ord = max(self.interp_ord - 1, 0)
        fes_T = [ngs.L2(self.mesh, order=scalar_ord, dgjumps=self.DG) for _ in self.ENERGY_COMPONENTS]
        # Insert right after alpha_c so the optional mean-zero-pressure NumberSpace stays last (it is accessed as
        # U[-1]/V[-1] in the TwoFluidModel forms).
        n = self.model_components['t_c']
        spaces[n:n] = fes_T
        return FESpace(spaces, dgjumps=self.DG)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _set_model_parameters(self) -> None:
        super()._set_model_parameters()

        model_config = self.model_functions.config
        if not model_config.has_section('ENERGY'):
            raise ValueError('TwoFluidModelEnergy requires an [ENERGY] section in model_dir/model_config.')

        allowed = {'cp_c', 'cp_d', 'k_c', 'k_d', 't_sat', 't_backflow', 'epsilon', 'interphase_heat_transfer'}
        unknown = set(model_config['ENERGY']) - allowed
        if unknown:
            raise ValueError('Unknown model [ENERGY] option(s): {}.'.format(', '.join(sorted(unknown))))
        for required in ('cp_c', 'cp_d', 'k_c', 'k_d', 't_sat'):
            if not model_config.has_option('ENERGY', required):
                raise ValueError("model_dir/model_config [ENERGY] requires '{}'.".format(required))

        def get(key, default=None):
            return float(model_config.get('ENERGY', key, fallback=default))

        self.cp_c, self.cp_d = get('cp_c'), get('cp_d')
        self.k_c, self.k_d = get('k_c'), get('k_d')
        self.T_sat = get('t_sat')
        self.T_backflow = get('t_backflow', self.T_sat)
        self.energy_epsilon = get('epsilon', 1e-3)
        self.interphase_heat_transfer = model_config.get('ENERGY', 'interphase_heat_transfer',
                                                         fallback='RanzMarshall').strip()
        if self.interphase_heat_transfer not in self.INTERPHASE_HEAT_TRANSFER_MODELS:
            raise ValueError("[ENERGY] interphase_heat_transfer must be one of: {}."
                             .format(', '.join(self.INTERPHASE_HEAT_TRANSFER_MODELS)))

        self.gas_free = model_config.has_section('GAS_FREE')
        if self.gas_free:
            unknown = set(model_config['GAS_FREE']) - self.GAS_FREE_OPTIONS
            if unknown:
                raise ValueError('Unknown model [GAS_FREE] option(s): {}.'.format(', '.join(sorted(unknown))))
            self.gas_free_alpha_min = float(model_config.get('GAS_FREE', 'alpha_min', fallback='1e-3'))
            self.gas_free_relaxation_time = float(model_config.get('GAS_FREE', 'relaxation_time', fallback='1e-4'))
            drift = model_config.get('GAS_FREE', 'drift_velocity', fallback='terminal').strip().lower()
            self._gas_free_drift_setting = 'terminal' if drift == 'terminal' else float(drift)
            if not (0.0 < self.gas_free_alpha_min < 1.0 and self.gas_free_relaxation_time > 0.0):
                raise ValueError('[GAS_FREE] needs 0 < alpha_min < 1 and relaxation_time > 0.')
            # Called again every time step (update_model_variables): keep the Parameter the forms refer to.
            if not hasattr(self, '_gas_free_drift'):
                self._gas_free_drift = ngs.Parameter(0.0)
                self.gas_free_drift_velocity = None      # U_r [m/s], set when the forms are built

    # ------------------------------------------------------------------
    # Gas velocity where there is (almost) no gas ([GAS_FREE])
    # ------------------------------------------------------------------

    @staticmethod
    def gas_free_weight(alpha_d, alpha_min: float):
        """ w(alpha_d) = (1 - min(alpha_d/alpha_min, 1))^2, clipped to 1 for alpha_d <= 0. """
        s = Min(Max(alpha_d / alpha_min, 0.0), 1.0)
        return (1.0 - s) * (1.0 - s)

    def _gas_free_coefficients(self, time_step: int):
        """ (K_d, K_c): the rate of the pull on u_d (per unit gas mass) and the liquid's reaction. """
        ts = time_step
        Ac = self.UIter.components[self.model_components['alpha_c']]
        Ad = 1 - Ac
        m = self.C_VM[ts] * self.rho_c[ts] / self.rho_d[ts] if self.VM_switch else 0.0
        K_d = self.gas_free_weight(Ad, self.gas_free_alpha_min) * (1.0 + m) / self.gas_free_relaxation_time
        # K_d vanishes where alpha_c <= 1 - alpha_min, so bounding alpha_c there only avoids 0/0 in pure gas.
        K_c = -Ad * self.rho_d[ts] / (Max(Ac, 1.0 - self.gas_free_alpha_min) * self.rho_c[ts]) * K_d
        return K_d, K_c

    def terminal_rise_velocity(self) -> float:
        """
        Terminal rise velocity of a single bubble relative to the liquid with the model's own drag closure: the slip
        U_r at which the drag on the gas (per unit gas mass) balances buoyancy,
            0.75 C_D(Re) rho_c / (rho_d dp) U_r^2 = (rho_c - rho_d) / rho_d |g|,   Re = U_r dp / nu_c,
        i.e. U_r = sqrt(4 |g| dp (rho_c - rho_d) / (3 C_D rho_c)), solved by damped fixed-point iteration.
        """
        if not self.drag_switch:
            raise ValueError('[GAS_FREE] drift_velocity = terminal needs a drag model ([TFM] IME drag).')
        dim = self.mesh.dim
        el = next(iter(self.mesh.Elements(ngs.VOL)))
        centre = [sum(self.mesh[v].point[i] for v in el.vertices) / len(el.vertices) for i in range(dim)]
        mip = self.mesh(*centre)

        def value(cf):
            return float(ngs.CoefficientFunction(cf)(mip))

        rho_c, rho_d, dp = value(self.rho_c[0]), value(self.rho_d[0]), value(self.dp[0])
        g = value(ngs.Norm(self.gravity))
        zero = ngs.CoefficientFunction(tuple([0.0] * dim))
        u = 0.1
        for _ in range(500):
            wd = ngs.CoefficientFunction(tuple([u] + [0.0] * (dim - 1)))
            Cd = value(self._get_drag_coeff(wd, zero, ngs.CoefficientFunction(0.0), 0))
            u_new = math.sqrt(4.0 * g * dp * (rho_c - rho_d) / (3.0 * Cd * rho_c))
            if abs(u_new - u) <= 1e-12 * u_new:
                return u_new
            u = 0.5 * (u + u_new)
        raise RuntimeError('The terminal rise velocity iteration did not converge.')

    def _gas_free_drift_vector(self):
        """ U_r e_up, with U_r evaluated once when the forms are first built (the drag closure is set by then). """
        if self.gas_free_drift_velocity is None:
            setting = self._gas_free_drift_setting
            self.gas_free_drift_velocity = self.terminal_rise_velocity() if setting == 'terminal' else setting
            self._gas_free_drift.Set(self.gas_free_drift_velocity)
            logging.info('[GAS_FREE] u_d is pulled towards u_c + %.4g m/s up where alpha_d < %g (relaxation time %g s).',
                         self.gas_free_drift_velocity, self.gas_free_alpha_min, self.gas_free_relaxation_time)
        return self._gas_free_drift * (-self.gravity / ngs.Norm(self.gravity))

    def _gas_sources(self) -> List:
        """ (volumetric gas source rate [1/s], dx restricted to its region) for every gas source. """
        sources = []
        if self.injection_switch and self.inj_mass_flowrate:
            sources.append((self.inj_mass_flowrate,
                            ngs.dx(definedon=self.mesh.Materials(self.injection_region))))
        return sources

    def _interphase_heat_transfer_coefficient(self, time_step: int):
        """ h_i * a_i [W/m^3/K], lagged on the Picard iterate. """
        if self.interphase_heat_transfer == 'none':
            return None
        comp = self.model_components
        ts = time_step
        wc = self.UIter.components[comp['u_c']]
        wd = self.UIter.components[comp['u_d']]
        Ad = 1 - self.UIter.components[comp['alpha_c']]
        dp, nu_c, rho_c = self.dp[ts], self.nu_c[ts], self.rho_c[ts]

        # Ranz-Marshall: Nu = 2 + 0.6 Re^1/2 Pr^1/3, interfacial area density a_i = 6 A_d / dp.
        Re = ngs.Norm(wd - wc) * dp / nu_c
        Pr = nu_c * rho_c * self.cp_c / self.k_c
        Nu = 2.0 + 0.6 * ngs.sqrt(Re) * Pr ** (1.0 / 3.0)
        a_i = 6.0 * Max(Ad, ngs.CoefficientFunction(0.0)) / dp
        return Nu * self.k_c / dp * a_i

    def _energy_phases(self, U, V, time_step: int):
        """ Per-phase quantities used by the energy forms. """
        comp = self.model_components
        ts = time_step
        Ac = self.UIter.components[comp['alpha_c']]
        return [
            # (phase, T, test, lagged fraction, wind, rho*cp, conductivity, velocity variable)
            ('c', U[comp['t_c']], V[comp['t_c']], Ac, self.UIter.components[comp['u_c']],
             self.rho_c[ts] * self.cp_c, self.k_c, 'u_c'),
            ('d', U[comp['t_d']], V[comp['t_d']], 1 - Ac, self.UIter.components[comp['u_d']],
             self.rho_d[ts] * self.cp_d, self.k_d, 'u_d'),
        ]

    # ------------------------------------------------------------------
    # Weak forms
    # ------------------------------------------------------------------

    def time_derivative_terms(self, gfu_lst: List[List[GridFunction]], scheme: str, step: int = 1):
        """ Adds the phase-fraction weighted storage terms (A_k^n + eps)(T_k - T_k^n) of the energy equations. """
        a, L = super().time_derivative_terms(gfu_lst, scheme, step)
        if scheme not in ('implicit euler', 'crank nicolson'):
            raise ValueError('TwoFluidModelEnergy only supports the implicit euler and crank nicolson schemes, '
                             'not "{}".'.format(scheme))

        U, V = self.get_trial_and_test_functions()
        comp = self.model_components
        eps = self.energy_epsilon
        Ac_old = gfu_lst[1][comp['alpha_c']]
        for name, A_old in (('t_c', Ac_old), ('t_d', 1 - Ac_old)):
            i = comp[name]
            a[0] += (A_old + eps) * U[i] * V[i]
            L[0] += (A_old + eps) * gfu_lst[1][i] * V[i]
        return a, L

    def construct_bilinear_time_ODE(self,
                                    U: Union[List[ProxyFunction], List[GridFunction]],
                                    V: List[ProxyFunction],
                                    dt: Parameter = Parameter(1.0),
                                    time_step: int = 0) -> List:
        a = super().construct_bilinear_time_ODE(U, V, dt, time_step)[0]
        n = self.n
        pen = self.penalty_interior
        eps = self.energy_epsilon
        zero = ngs.CoefficientFunction(0.0)
        phases = self._energy_phases(U, V, time_step)

        for (phase, T, s, A, w, rho_cp, k, u_var) in phases:
            # Advection A w.grad(T) = C(A T; s) - C(A; T s), with C(phi; psi) the upwind DG form of div(w phi)
            # tested with psi:  C = -(phi w.grad(psi))_K + (F(phi) jump(psi))_F,  F = upwind flux.
            # The volume terms combine to (A w.grad(T) s). Other() cannot be applied to a product of a field
            # and a trial function, so the fluxes are built from the separate traces.
            wn = w * n
            AT, AT_other = A * T, A.Other() * T.Other()
            flux_AT = 0.5 * (AT + AT_other) * wn + 0.5 * ngs.Norm(wn) * (AT - AT_other)
            flux_A = 0.5 * (A + A.Other()) * wn + 0.5 * ngs.Norm(wn) * (A - A.Other())
            a += (dt * A * (w * ngs.grad(T)) * s) * ngs.dx
            a += (dt * (flux_AT * jump(s) - flux_A * (T * s - T.Other() * s.Other()))) * ngs.dx(skeleton=True)
            open_reg = self._bc_regex('zero_stress', u_var)
            if open_reg and phase == 'c':
                # Outflow terms cancel; backflow enters as pure liquid at T_backflow (linear part in
                # construct_linear): -min(w.n, 0) (T_backflow - T) s. No gas enters (A_d = 0 in backflow).
                a += (dt * -Min(wn, zero) * T * s) * self._ds(open_reg)

            # Diffusion div((A + eps) kappa grad T), SIPG. Walls are adiabatic (natural BC).
            D = (A + eps) * (k / rho_cp)
            a += (dt * D * ngs.grad(T) * ngs.grad(s)) * ngs.dx
            a += (dt * (-(n * weighted_grad_avg(s, D)) * jump(T)
                        - (n * weighted_grad_avg(T, D)) * jump(s)
                        + avg(D) * pen * jump(T) * jump(s))) * ngs.dx(skeleton=True)

        # Gas created by the gas sources: mdot (T_sat - T_d), T_sat part in construct_linear.
        comp = self.model_components
        Td_trial, sd = U[comp['t_d']], V[comp['t_d']]
        for mdot, dx_src in self._gas_sources():
            a += (dt * mdot * Td_trial * sd) * dx_src

        # Interphase sensible heat exchange, implicit in both temperatures.
        H = self._interphase_heat_transfer_coefficient(time_step)
        if H is not None:
            (_, Tc, sc, _, _, rho_cp_c, _, _), (_, Td, sd, _, _, rho_cp_d, _, _) = phases
            a += (dt * H / rho_cp_c * (Tc - Td) * sc) * ngs.dx
            a += (dt * H / rho_cp_d * (Td - Tc) * sd) * ngs.dx

        # [GAS_FREE] pull of u_d towards u_c + U_r e_up: the u_d - u_c part (U_r part in construct_linear).
        if self.gas_free:
            uc, ud = U[comp['u_c']], U[comp['u_d']]
            vc, vd = V[comp['u_c']], V[comp['u_d']]
            K_d, K_c = self._gas_free_coefficients(time_step)
            a += (dt * K_d * (ud - uc) * vd) * ngs.dx
            a += (dt * K_c * (ud - uc) * vc) * ngs.dx

        return [a]

    def construct_linear(self,
                         V: List[ProxyFunction],
                         gfu_0: Optional[List[GridFunction]],
                         dt: Parameter,
                         time_step: int) -> List:
        L = super().construct_linear(V, gfu_0, dt, time_step)[0]
        comp = self.model_components
        n = self.n
        zero = ngs.CoefficientFunction(0.0)
        U, _ = self.get_trial_and_test_functions()

        for (phase, _, s, _, w, rho_cp, _, u_var) in self._energy_phases(U, V, time_step):
            # Backflow through open boundaries enters as pure liquid (as for alpha_c) at T_backflow.
            open_reg = self._bc_regex('zero_stress', u_var)
            if open_reg and phase == 'c':
                L += (dt * -s * self.T_backflow * Min(w * n, zero)) * self._ds(open_reg)

            # Prescribed heat fluxes [W/m^2] into the domain.
            name = 't_' + phase
            for marker, val_list in self.BC.get('neumann', {}).get(name, {}).items():
                L += (dt * val_list[time_step] / rho_cp * s) * self._ds(marker)

        # Gas created by the gas sources enters at T_sat: mdot (T_sat - T_d), T_d part in the bilinear form.
        sd = V[comp['t_d']]
        for mdot, dx_src in self._gas_sources():
            L += (dt * mdot * self.T_sat * sd) * dx_src

        # [GAS_FREE] pull of u_d towards u_c + U_r e_up: the U_r part (u_d - u_c part in the bilinear form).
        if self.gas_free:
            vc, vd = V[comp['u_c']], V[comp['u_d']]
            K_d, K_c = self._gas_free_coefficients(time_step)
            drift = self._gas_free_drift_vector()
            L += (dt * K_d * drift * vd) * ngs.dx
            L += (dt * K_c * drift * vc) * ngs.dx

        return [L]
