import numpy as np
from typing import List, Tuple, Dict, Optional
import itertools
from scipy import constants, optimize
from sympy.polys.numberfields.basis import nilradical_mod_p

from generic_configuration_gibbs_free_energy import GenericConfiguration, GeometrySettings, GeometryType


class GibbsEnergyCalculator:
    """
    Class to calculate the Gibbs free energy of a BNP based on a given configuration.
    """

    def __init__(self, config: GenericConfiguration):
        """
        Initializes the calculator with a specific configuration.
        """
        self.config = config

        # Extract raw data for easier access
        self.ab_data = config.ab_data.data
        self.surface_energy = config.surface_energy_data
        self.pure_mat_data = [config.a_data.data, config.b_data.data]


    def calculate_total_Gibbs_free_energy(self,
                                          ratio_A_alpha: float, ratio_B_alpha: float,
                                          n_total: float, xB_total: float,
                                          T: float,
                                          phase_pair: Tuple[str, str],
                                          geometry: GeometrySettings) -> float:
        """
        Calculates the total Gibbs free energy for a specific split of phases.

        Args:
            ratio_A_alpha: Fraction of total A moles that are in phase alpha (0 to 1).
            ratio_B_alpha: Fraction of total B moles that are in phase alpha (0 to 1).
            n_total: Total number of moles in the system.
            xB_total: Total mole fraction of B in the system.
            T: Temperature.
            phase_pair: Names of the two phases (alpha, beta).
            geometry: Geometric settings.

        Returns:
            Total Gibbs free energy (G_total).
        """

        n_mp, x_mp = self.calc_n_x(ratio_A_alpha, ratio_B_alpha,
                                   n_total, xB_total,
                                   T, phase_pair,
                                   geometry)
        if np.any(n_mp <= 0) or np.any(x_mp < 0) or np.any(np.isnan(n_mp)) or np.any(np.isnan(x_mp)):
            return 1.0
        elif geometry.has_outer_shell and n_total - np.sum(n_mp) <= 0:
            return 1.0
        elif np.sum(x_mp[:,0]) < 0.1 or np.sum(x_mp[:,1]) < 0.1:
            return 1.0

        G_ideal_total = self.calc_G_ideal(n_mp, x_mp, n_total, T, phase_pair, geometry)
        G_excess_total = self.calc_G_excess(n_mp, x_mp, T, phase_pair)
        try:
            G_surface_total = self.calc_G_surface(n_mp, x_mp,
                                             T, phase_pair,
                                             geometry)
        except ValueError:
            return 1.0
        G_total = G_ideal_total + G_excess_total + G_surface_total


        return G_total

    def calc_G_surface(self, n_mp, x_mp,
                       T, phase_pair,
                       geometry: GeometrySettings
                       ):

        AB_names = [mat.name for mat in self.pure_mat_data]
        v_mp = self.calc_v_mp(T, phase_pair)
        xv23_mp = (v_mp ** (2 / 3)) * x_mp

        def helper_calc_sigma_interface(phase_alpha, xv23_alpha, phase_beta="Vacuum", xv23_beta=None):
            sigma_interface = 0
            denominator = np.sum(xv23_alpha)


            if phase_beta == "Vacuum":
                for i, mat_alpha_name in enumerate(AB_names):
                    # Access: AaBb[Mat][Phase]['Vacuum'](T)
                    sigma_func = self.surface_energy.AaBb[mat_alpha_name][phase_alpha]['Vacuum']
                    sigma_interface += xv23_alpha[i] * sigma_func(T)
            else:
                denominator *= np.sum(xv23_beta)
                for i, mat_alpha_name in enumerate(AB_names):
                    for j, mat_beta_name in enumerate(AB_names):
                        # Access: AaBb[MatA][PhaseA][MatB][PhaseB](T)
                        sigma_func = self.surface_energy.AaBb[mat_alpha_name][phase_alpha][mat_beta_name][phase_beta]
                        sigma_interface += xv23_alpha[i] * xv23_beta[j] * sigma_func(T)

            sigma_interface /= denominator if denominator > 1e-15 else 1e9
            return sigma_interface

        def helper_calc_surface_area_sphere(V):
            r_core = ((3 * V) / (4 * np.pi)) ** (1 / 3)
            A_core_shell = 4 * np.pi * r_core ** 2
            return A_core_shell

        def helper_calc_spheric_Janus_a_h(nv_mp):
            V_total = np.sum(nv_mp)
            r_total = ((3 * V_total) / (4 * np.pi)) ** (1 / 3)
            h_alpha_coeff = [-np.pi/3, np.pi*r_total, 0, -np.sum(nv_mp[:,0])]
            h_alpha_roots = np.roots(h_alpha_coeff)
            real_h_alpha_roots = h_alpha_roots[np.isreal(h_alpha_roots)].real
            valid_h_alpha_roots = real_h_alpha_roots[(real_h_alpha_roots > 0) & (real_h_alpha_roots < 2 * r_total)]

            return float(valid_h_alpha_roots)

        def helper_get_outer_shell_xv23():
            x_m_outer_layer = np.array([1 - geometry.shell.material, geometry.shell.material])
            v_outer_layer = self.calc_v_mp(T, (geometry.shell.phase, geometry.shell.phase))
            xv23_outer_layer = (v_outer_layer[:, 0] ** (2 / 3)) * x_m_outer_layer
            return xv23_outer_layer

        def helper_calc_outer_shell_vacuum_energy():
            r_no_outer_layer = ((3 * np.sum(n_mp * v_mp)) / (4 * np.pi)) ** (1 / 3)
            shell_mat_name = AB_names[geometry.shell.material]
            r_with_outer_layer = r_no_outer_layer + self.pure_mat_data[geometry.shell.material].atomic_radius
            A_outer_layer_vacuum = 4 * np.pi * (r_with_outer_layer ** 2)
            surface_energy_outer_layer_vacuum = self.surface_energy.AaBb[shell_mat_name][geometry.shell.phase]['Vacuum'](T) * A_outer_layer_vacuum
            return surface_energy_outer_layer_vacuum

        match (geometry.geometry_type, geometry.has_outer_shell):
            case (GeometryType.CORE_SHELL, False):
                A_core_shell = helper_calc_surface_area_sphere(np.sum(v_mp[:, 0] * n_mp[:, 0]))
                sigma_core_shell_interface = helper_calc_sigma_interface(phase_pair[0], xv23_mp[:,0], phase_pair[1], xv23_mp[:,1])
                surface_energy_core_to_shell = sigma_core_shell_interface * A_core_shell
                A_total = helper_calc_surface_area_sphere(np.sum(v_mp * n_mp))
                sigma_shell_vacuum_interface = helper_calc_sigma_interface(phase_pair[1], xv23_mp[:,1])
                surface_energy_shell_to_vacuum = sigma_shell_vacuum_interface * A_total
                total_surface_energy = surface_energy_core_to_shell + surface_energy_shell_to_vacuum
                return total_surface_energy

            case (GeometryType.JANUS, False):
                sigma_Janus_interface = helper_calc_sigma_interface(phase_pair[0], xv23_mp[:,0], phase_pair[1], xv23_mp[:,1])
                sigma_Janus_alpha_vacuum = helper_calc_sigma_interface(phase_pair[0], xv23_mp[:,0])
                sigma_Janus_beta_vacuum = helper_calc_sigma_interface(phase_pair[1], xv23_mp[:,1])
                nv_mp = n_mp * v_mp
                def helper_calc_Janus_geo(sigma_Janus_interface, sigma_Janus_alpha_vacuum, sigma_Janus_beta_vacuum):
                    h_alpha_initial_guess = helper_calc_spheric_Janus_a_h(nv_mp)
                    h_beta_initial_guess = 2 * (((3 * np.sum(nv_mp)) / (4 * np.pi)) ** (1/3)) - h_alpha_initial_guess
                    h_initial_guess = np.array([h_alpha_initial_guess, h_beta_initial_guess])
                    def helper_Janus_geo_ri_calc(h_alpha_beta):
                        r_a, r_b = h_alpha_beta
                        r2_a = r_a ** 2
                        r2_b = r_b ** 2
                        s_i = sigma_Janus_interface
                        s_a = sigma_Janus_alpha_vacuum
                        s_b = sigma_Janus_beta_vacuum
                        ri2_calc_coeff = [
                            s_a + s_b + s_i,
                            (r2_a + r2_b) * s_i - (s_a - s_b) * (r2_a - r2_b) * s_i,
                            r2_a * r2_b * (s_i - s_a - s_b)
                        ]
                        ri2_vals = np.roots(ri2_calc_coeff)
                        ri2_vals = ri2_vals[np.isreal(ri2_vals)].real
                        ri2_vals = ri2_vals[ri2_vals > 0]
                        if len(ri2_vals) == 0:
                            return [1e5, 1e5]
                        ri2_vals = np.min(ri2_vals)
                        r_i = ri2_vals ** (1/2)

                        eq1 = ((1/6) * np.pi * r_a) * (3 * (r_i ** 2) + r_a ** 2) - np.sum(nv_mp[:,0])
                        eq2 = ((1/6) * np.pi * r_b) * (3 * (r_i ** 2) + r_b ** 2) - np.sum(nv_mp[:,1])

                        return [eq1, eq2]
                    sol = optimize.root(helper_Janus_geo_ri_calc, h_initial_guess, method='hybr')
                    term_inside_sqrt = (((6 * np.sum(nv_mp[:,0])) / (np.pi * sol.x[0])) - sol.x[0] ** 2) / 3
                    if term_inside_sqrt < 0:
                        raise ValueError("Unphysical Janus Geometry")
                    a_i_result = (term_inside_sqrt ** (1/2))
                    return sol.x[0], sol.x[1], a_i_result
                h_alpha, h_beta, a_i = helper_calc_Janus_geo(sigma_Janus_interface, sigma_Janus_alpha_vacuum, sigma_Janus_beta_vacuum)
                surface_energy_alpha = sigma_Janus_alpha_vacuum * np.pi * (h_alpha ** 2 + a_i ** 2)
                surface_energy_beta = sigma_Janus_beta_vacuum * np.pi * (h_beta ** 2 + a_i ** 2)
                surface_energy_interface = sigma_Janus_interface * np.pi * (a_i ** 2)
                surface_energy_total = surface_energy_alpha + surface_energy_beta + surface_energy_interface
                return surface_energy_total
            case (GeometryType.CORE_SHELL, True):
                # 1. Inner Interface (Alpha -> Beta)
                # Assumes Alpha is Core, Beta is surrounding Mantle
                A_core_shell = helper_calc_surface_area_sphere(np.sum(v_mp[:, 0] * n_mp[:, 0]))
                sigma_core_shell_interface = helper_calc_sigma_interface(phase_pair[0], xv23_mp[:,0], phase_pair[1], xv23_mp[:,1])
                surface_energy_core_to_shell = sigma_core_shell_interface * A_core_shell

                # 2. Middle Interface (Beta -> Outer Shell)
                # Construct properties for the pure outer shell material
                xv23_outer_layer = helper_get_outer_shell_xv23()
                sigma_shell_outer_layer = helper_calc_sigma_interface(phase_pair[1], xv23_mp[:,1], geometry.shell.phase, xv23_outer_layer)
                A_shell_to_outer_layer = helper_calc_surface_area_sphere(np.sum(n_mp * v_mp))
                surface_energy_shell_to_outer_layer = sigma_shell_outer_layer * A_shell_to_outer_layer

                # 3. Outer Interface (Outer Shell -> Vacuum)
                surface_energy_outer_layer_vacuum = helper_calc_outer_shell_vacuum_energy()
                total_surface_energy = surface_energy_core_to_shell + surface_energy_outer_layer_vacuum + surface_energy_shell_to_outer_layer
                return total_surface_energy

            case (GeometryType.SPHERIC_JANUS, True):
                # Calculate geometry of spheric Janus
                h_alpha = helper_calc_spheric_Janus_a_h(n_mp * v_mp)
                h_beta = float(2 * (((3 * np.sum(n_mp * v_mp)) / (4 * np.pi)) ** (1/3)) - h_alpha)
                term_inside_sqrt = (((6 * np.sum(n_mp[:,0] * v_mp[:,0])) / (np.pi * h_alpha)) - (h_alpha ** 2)) / 3
                if term_inside_sqrt < 0:
                    raise ValueError("Unphysical Spheric Janus Geometry")
                a_i = float(term_inside_sqrt ** (1/2))

                xv23_outer_layer = helper_get_outer_shell_xv23()

                # Alpha -> outer layer interface
                A_alpha_outer_layer = np.pi * (h_alpha ** 2 + a_i ** 2)
                sigma_alpha_outer_layer = helper_calc_sigma_interface(phase_pair[0], xv23_mp[:,0], geometry.shell.phase, xv23_outer_layer)
                surface_energy_alpha_outer_layer = sigma_alpha_outer_layer * A_alpha_outer_layer

                # Beta -> outer layer interface
                A_beta_outer_layer = np.pi * (h_beta ** 2 + a_i ** 2)
                sigma_beta_outer_layer = helper_calc_sigma_interface(phase_pair[1], xv23_mp[:,1], geometry.shell.phase, xv23_outer_layer)
                surface_energy_beta_outer_layer = sigma_beta_outer_layer * A_beta_outer_layer

                # Alpha -> Beta interface
                A_alpha_beta = np.pi * (a_i ** 2)
                sigma_alpha_beta = helper_calc_sigma_interface(phase_pair[0], xv23_mp[:,0], phase_pair[1], xv23_mp[:,1])
                surface_energy_alpha_beta = sigma_alpha_beta * A_alpha_beta

                # Outer layer -> vacuum interface
                surface_energy_outer_layer_vacuum = helper_calc_outer_shell_vacuum_energy()

                total_surface_energy = surface_energy_alpha_outer_layer + surface_energy_beta_outer_layer + surface_energy_outer_layer_vacuum + surface_energy_alpha_beta
                return total_surface_energy


    def calc_G_ideal(self, n_mp, x_mp,
                     n_total,
                     T, phase_pair,
                     geometry: GeometrySettings
                     ):

        """
        Calculates the Ideal Gibbs free energy including the mixing term and outer shell.
        """
        R = constants.gas_constant

        g0_ideal_mp = np.array([
            [mat.phases[p].g0(T) for p in phase_pair]
            for mat in self.pure_mat_data
        ])

        with np.errstate(divide='ignore', invalid='ignore'):
            # x*ln(x) term, handling the x=0 limit physically
            entropy_term = np.where(x_mp > 0, x_mp * np.log(x_mp), 0.0)

        g_ideal_p = np.sum(g0_ideal_mp * x_mp, axis=0) + R * T * np.sum( entropy_term, axis=0)
        G_ideal_p = (g_ideal_p * np.sum(n_mp, axis=0))
        G_ideal = np.sum(G_ideal_p)

        if geometry.has_outer_shell:
            g_outer_shell = self.pure_mat_data[geometry.shell.material].phases[geometry.shell.phase].g0(T)
            n_outer_shell = n_total - np.sum(n_mp)
            G_outer_shell = n_outer_shell * g_outer_shell
            G_ideal += G_outer_shell

        return G_ideal


    def calc_G_excess(self, n_mp, x_mp, T, phase_pair):

        L_ip = np.column_stack([
            self.ab_data.phases[p].get_Li_per_T(T)
            for p in phase_pair
        ])
        g_excess_p = np.array([0, 0])
        xAxB_p = np.prod(x_mp, axis=0)
        xA_minus_xB_p = -np.diff(x_mp, axis=0)

        for i in range(len(L_ip)):
            g_excess_p = g_excess_p + xAxB_p  * L_ip[i, :] * (xA_minus_xB_p ** i)

        G_excess_p = g_excess_p * np.sum(n_mp, axis=0)
        G_excess = np.sum(G_excess_p)
        return G_excess

    def calc_n_x(self, ratio_A_alpha, ratio_B_alpha,
                 n_total, xB_total,
                 T, phase_pair,
                 geometry):

        def helper_basic_split(ratio_A_alpha = ratio_A_alpha, ratio_B_alpha = ratio_B_alpha,
                               n_A_total_core = n_total * (1 - xB_total), n_B_total_core =n_total * xB_total):
            n_A_alpha = ratio_A_alpha * n_A_total_core
            n_B_alpha = ratio_B_alpha * n_B_total_core
            n_A_beta = n_A_total_core - n_A_alpha
            n_B_beta = n_B_total_core - n_B_alpha
            n_mp = np.array([
                [n_A_alpha, n_A_beta],
                [n_B_alpha, n_B_beta]
            ])

            x_A_alpha = n_A_alpha / (n_A_alpha + n_B_alpha) if np.sum(n_mp[:,0]) > n_total * 1e-6 else 0.0
            x_B_alpha = n_B_alpha / (n_A_alpha + n_B_alpha) if np.sum(n_mp[:,0]) > n_total * 1e-6 else 0.0
            x_A_beta = n_A_beta / (n_A_beta + n_B_beta) if np.sum(n_mp[:,1]) > n_total * 1e-6 else 0.0
            x_B_beta = n_B_beta / (n_A_beta + n_B_beta) if np.sum(n_mp[:,1]) > n_total * 1e-6 else 0.0
            x_mp = np.array([
                [x_A_alpha, x_A_beta],
                [x_B_alpha, x_B_beta]
            ])

            return n_mp, x_mp

        if geometry.has_outer_shell:

            n_mp, x_mp = helper_basic_split(ratio_A_alpha = ratio_A_alpha, ratio_B_alpha = ratio_B_alpha,
                                            n_A_total_core = n_total * (1 - xB_total), n_B_total_core =n_total * xB_total)
            v_mp = self.calc_v_mp(T, phase_pair)
            V_no_shell = np.sum(v_mp*n_mp)
            r_no_shell = ((3 * V_no_shell) / (4 * np.pi)) ** (1/3)
            V_shell = (4 / 3) * np.pi * ((r_no_shell + 2 * self.pure_mat_data[geometry.shell.material].atomic_radius) ** 3) - V_no_shell
            v_shell_mat = self.pure_mat_data[geometry.shell.material].phases[geometry.shell.phase].v(T)
            n_shell_prev = V_shell / v_shell_mat
            num_of_tries = 0
            n_A_before_shell = n_total * (1 - xB_total)
            n_B_before_shell = n_total * xB_total
            while True:
                num_of_tries += 1
                n_A_core = n_A_before_shell - n_shell_prev * (1 - geometry.shell.material)
                n_B_core = n_B_before_shell - n_shell_prev * geometry.shell.material
                if n_A_core <= 0 or n_B_core <= 0:
                    return np.zeros_like(n_mp), np.zeros_like(x_mp)

                n_mp, x_mp = helper_basic_split(ratio_A_alpha = ratio_A_alpha, ratio_B_alpha = ratio_B_alpha,
                                               n_A_total_core = n_A_core, n_B_total_core = n_B_core)
                V_no_shell = np.sum(v_mp*n_mp)
                r_no_shell = ((3 * V_no_shell) / (4 * np.pi)) ** (1/3)
                V_shell = (4 / 3) * np.pi * ((r_no_shell + self.pure_mat_data[geometry.shell.material].atomic_radius) ** 3) - V_no_shell
                n_shell_curr = V_shell / v_shell_mat
                tol = abs(n_shell_curr - n_shell_prev)
                # print(tol, num_of_tries)
                if tol < n_total*1e-6:
                    break
                elif num_of_tries > 100:
                    print("Couldn't converge on number of moles in outer shell")
                    break
                else:
                    n_shell_prev = 0.7 * n_shell_curr + 0.3 * n_shell_prev


        else:
            n_mp, x_mp = helper_basic_split(ratio_A_alpha = ratio_A_alpha, ratio_B_alpha = ratio_B_alpha,
                                            n_A_total_core = n_total * (1 - xB_total), n_B_total_core =n_total * xB_total)
        return n_mp, x_mp

    def calc_v_mp(self, T: float, phase_pair: Tuple[str, str]) -> np.ndarray:
        """
        Calculates the molar volume matrix for materials A and B in the given phases.
        """
        return np.array([
            [mat.phases[p].v(T) for p in phase_pair]
            for mat in self.pure_mat_data
        ])

    # ... existing code ...
    def calculate_single_phase_gibbs_free_energy(self,
                                                 n_total: float, xB_total: float,
                                                 T: float,
                                                 phase: str,
                                                 geometry: GeometrySettings) -> Tuple[float, float, float, float]:
        """
        Calculates the total Gibbs free energy for a single-phase particle (Core only),
        optionally with a pure outer shell.
        """

        # 1. Setup Base Moles
        n_A_total = n_total * (1 - xB_total)
        n_B_total = n_total * xB_total

        n_A_core = n_A_total
        n_B_core = n_B_total
        n_shell = 0.0

        # Pre-fetch Molar Volumes for the core phase
        v_A = self.pure_mat_data[0].phases[phase].v(T)
        v_B = self.pure_mat_data[1].phases[phase].v(T)

        # 2. Handle Outer Shell (Calculate n_shell and adjust core moles)
        if geometry.has_outer_shell:

            shell_mat_idx = geometry.shell.material
            shell_phase = geometry.shell.phase

            # Get shell properties
            v_shell_pure = self.pure_mat_data[shell_mat_idx].phases[shell_phase].v(T)
            r_atom_shell = self.pure_mat_data[shell_mat_idx].atomic_radius

            # Define function: F(n_shell) = V_calculated_from_geometry - V_calculated_from_moles
            # We want to find n_shell where F(n_shell) = 0
            max_shell_moles = n_A_total if shell_mat_idx == 0 else n_B_total
            if max_shell_moles < 1e-25:
                raise ValueError("Not enough material to form the requested outer shell.")

            V_core_initial = v_A * n_A_core + v_B * n_B_core
            r_core = ((3 * V_core_initial) / (4 * np.pi)) ** (1 / 3)
            V_shell = 4 * np.pi * (r_core ** 2) * r_atom_shell * 2
            n_shell_prev = V_shell / v_shell_pure
            num_of_tries = 0
            n_A_before_shell = n_total * (1 - xB_total)
            n_B_before_shell = n_total * xB_total

            while True:
                num_of_tries += 1
                n_A_core = n_A_before_shell - n_shell_prev * (1 - geometry.shell.material)
                n_B_core = n_B_before_shell - n_shell_prev * geometry.shell.material
                if n_A_core <= 0 or n_B_core <= 0:
                    return 1.0, 1.0, 1.0, 1.0

                V_no_shell = n_A_core * v_A + n_B_core * v_B
                r_no_shell = ((3 * V_no_shell) / (4 * np.pi)) ** (1 / 3)
                V_shell = (4 / 3) * np.pi * ((r_no_shell + 2*r_atom_shell) ** 3) - V_no_shell
                n_shell_curr = V_shell / v_shell_pure
                tol = abs(n_shell_curr - n_shell_prev)
                # print(tol, num_of_tries)
                if tol < n_total * 1e-6:
                    break
                elif num_of_tries > 100:
                    print("Couldn't converge on number of moles in outer shell")
                    break
                else:
                    n_shell_prev = 0.7 * n_shell_curr + 0.3 * n_shell_prev

            n_shell = n_shell_curr
            if shell_mat_idx == 0:
                n_A_core = n_total * (1 - xB_total) - n_shell
                n_B_core = n_total * xB_total
            else:
                n_A_core = n_total * (1 - xB_total)
                n_B_core = n_total * xB_total - n_shell

        # 3. Core Composition
        n_core_total = n_A_core + n_B_core
        if n_core_total <= 1e-25:  # Effectively zero/impossible
            return 0.0, 0.0, 0.0, 0.0

        x_A = n_A_core / n_core_total
        x_B = n_B_core / n_core_total

        # 4. Ideal Energy (Core)
        R = constants.gas_constant
        g0_A = self.pure_mat_data[0].phases[phase].g0(T)
        g0_B = self.pure_mat_data[1].phases[phase].g0(T)

        # Entropy term x*log(x)
        entropy = 0.0
        if x_A > 1e-12: entropy += x_A * np.log(x_A)
        if x_B > 1e-12: entropy += x_B * np.log(x_B)

        G_ideal = n_core_total * (x_A * g0_A + x_B * g0_B + R * T * entropy)

        # 4b. Ideal Energy (Shell)
        if geometry.has_outer_shell:
            g0_shell = self.pure_mat_data[geometry.shell.material].phases[geometry.shell.phase].g0(T)
            G_ideal += n_shell * g0_shell

        # 5. Excess Energy (Core)
        L_values = self.ab_data.phases[phase].get_Li_per_T(T)
        # Sum( L_i * (xA - xB)^i )
        interaction_sum = 0.0
        x_diff = x_A - x_B
        for i, L in enumerate(L_values):
            interaction_sum += L * (x_diff ** i)

        G_excess = n_core_total * x_A * x_B * interaction_sum

        # 6. Surface Energy
        AB_names = [mat.name for mat in self.pure_mat_data]
        mat_A_name, mat_B_name = AB_names[0], AB_names[1]

        # Re-calculate core geometry for final values
        V_core = n_A_core * v_A + n_B_core * v_B
        r_core = ((3 * V_core) / (4 * np.pi)) ** (1 / 3)
        A_core = 4 * np.pi * (r_core ** 2)

        G_surface = 0.0

        if geometry.has_outer_shell:
            # Core -> Shell Interface
            shell_name = AB_names[geometry.shell.material]
            shell_phase = geometry.shell.phase

            # Sigma = xA * sigma(A_core, Shell) + xB * sigma(B_core, Shell)
            sigma_A_shell = self.surface_energy.AaBb[mat_A_name][phase][shell_name][shell_phase](T)
            sigma_B_shell = self.surface_energy.AaBb[mat_B_name][phase][shell_name][shell_phase](T)
            sigma_core_shell =(((v_A ** (2/3)) * x_A) * sigma_A_shell + ((v_B ** (2/3)) * x_B) * sigma_B_shell) / ((v_A ** (2/3)) * x_A + (v_B ** (2/3)) * x_B)

            G_surface += A_core * sigma_core_shell

            # Shell -> Vacuum Interface
            r_total = r_core + self.pure_mat_data[geometry.shell.material].atomic_radius
            A_outer = 4 * np.pi * (r_total ** 2)
            sigma_shell_vac = self.surface_energy.AaBb[shell_name][shell_phase]['Vacuum'](T)

            G_surface += A_outer * sigma_shell_vac

        else:
            # Core -> Vacuum Interface
            sigma_A_vac = self.surface_energy.AaBb[mat_A_name][phase]['Vacuum'](T)
            sigma_B_vac = self.surface_energy.AaBb[mat_B_name][phase]['Vacuum'](T)
            sigma_core_vac =(((v_A ** (2/3)) * x_A) * sigma_A_vac + ((v_B ** (2/3)) * x_B) * sigma_B_vac) / ((v_A ** (2/3)) * x_A + (v_B ** (2/3)) * x_B)

            G_surface += A_core * sigma_core_vac

        return G_ideal + G_excess + G_surface, n_core_total, n_shell, x_B