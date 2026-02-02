import numpy as np
from typing import List, Tuple, Dict, Any
from system_data import SystemData
from scipy import optimize
class GibbsEnergyCalculator3Phase:
    """
    Calculates the total Gibbs free energy for a 3-phase system.
    Designed to accept a generic list of optimization variables.
    """
    def __init__(self, system_data: SystemData):
        self.system_data = system_data

    def calculate_total_energy(self, 
                               A_ratio_alpha: float, 
                               B_ratio_alpha: float,
                               T: float, 
                               n_total: float, 
                               xB_total: float, 
                               primary_phases: Tuple[str, str],
                               geometry_type: str,
                               has_skin: bool = False,
                               xB_skin: float = 1.0,
                            ) -> float:
        try:
            if has_skin:
                phases = primary_phases + ("Liquid",)
            else:
                phases = primary_phases
            

            n_mp, x_mp, r_vals = self._calc_mole_splits_and_geo(
                A_ratio_alpha, 
                B_ratio_alpha,
                T, 
                n_total, 
                xB_total, 
                phases,
                geometry_type,
                has_skin,
                xB_skin
            )
            G_ideal = self.calc_G_ideal(n_mp, x_mp, T, phases)
            G_excess = self.calc_G_excess(n_mp, x_mp, T, phases)
            G_surface = self.calc_G_surface(n_mp, x_mp, r_vals, T, phases, geometry_type, has_skin)

            return G_ideal + G_excess + G_surface
        
        except ValueError:
            # This catches failures from _calc_mole_splits_and_geo,
            # such as convergence failure or insufficient moles for the skin.
            return 1.0

    def calc_G_excess(self, n_mp: np.ndarray, x_mp: np.ndarray, T: float, phases: Tuple[str, ...]) -> float:
        """
        Calculates the excess Gibbs free energy for the system.
        n_mp: Mole matrix (rows: materials, columns: phases)
        x_mp: Mole fraction matrix (rows: materials, columns: phases)
        T: Temperature
        Returns the excess Gibbs free energy.
        """
        G_excess = 0.0
        num_phases = n_mp.shape[1]

        # 1. Get Interaction Data (e.g. "AgCu")
        mat_names = self.system_data.config.materials
        interaction_name = "".join(mat_names)
        
        interaction = self.system_data.get_interaction(interaction_name)

        # 2. Sum Excess Energy for each phase
        for i in range(num_phases):
            phase_name = phases[i]
            
            n_phase = np.sum(n_mp[:, i])

            L_values = interaction.phases[phase_name].get_Li_per_T(T)
            xA = x_mp[0, i]
            xB = x_mp[1, i]
            
            # Redlich-Kister: Sum( L_k * (xA - xB)^k )
            interaction_sum = np.sum([L * ((xA - xB) ** k) for k, L in enumerate(L_values)])
            G_excess += n_phase * xA * xB * interaction_sum

        return G_excess
        
    def calc_G_surface(
            self, n_mp: np.ndarray, 
            x_mp: np.ndarray, 
            r_vals: np.ndarray, 
            T: float, phases: Tuple[str, ...], 
            geometry_type: str, 
            has_skin: bool = False) -> float:
        """
        Calculates the surface Gibbs free energy.
        Uses r_vals which contains pre-calculated geometry parameters.
        """
        G_surface = 0.0
        v_mp = self._get_v_mp(T, phases)
        
        if geometry_type == "Janus":
            h_a, h_b, a_i = r_vals
            
            # Surface areas of the spherical caps (Alpha and Beta)
            # Area = pi * (h^2 + a^2)
            A_alpha_outer = np.pi * (h_a**2 + a_i**2)
            A_beta_outer = np.pi * (h_b**2 + a_i**2)
            A_interface = np.pi * a_i**2
            
            # 1. Alpha-Beta Interface Energy
            sigma_ab = self._calc_sigma((phases[0], phases[1]), T, v_mp[:,0], x_mp[:,0], v_mp[:,1], x_mp[:,1])
            G_surface += A_interface * sigma_ab
            
            if has_skin:
                # 2. Alpha-Skin & Beta-Skin Interfaces
                sigma_as = self._calc_sigma((phases[0], phases[2]), T, v_mp[:,0], x_mp[:,0], v_mp[:,2], x_mp[:,2])
                G_surface += A_alpha_outer * sigma_as
                
                sigma_bs = self._calc_sigma((phases[1], phases[2]), T, v_mp[:,1], x_mp[:,1], v_mp[:,2], x_mp[:,2])
                G_surface += A_beta_outer * sigma_bs
                
                # 3. Skin-Vacuum Interface
                mats = self.system_data.config.materials
                r_atom_A = self.system_data.material_data[mats[0]].atomic_radius
                r_atom_B = self.system_data.material_data[mats[1]].atomic_radius
                xB_skin = x_mp[1, 2]
                weighted_skin_thickness = 2 * (1 - xB_skin) * r_atom_A + 2 * xB_skin * r_atom_B

                r_a = (h_a**2 + a_i**2) / (2 * h_a)
                r_b = (h_b**2 + a_i**2) / (2 * h_b)

                A_alpha_vac = A_alpha_outer * ((r_a + weighted_skin_thickness) / r_a)**2
                A_beta_vac = A_beta_outer * ((r_b + weighted_skin_thickness) / r_b)**2
                A_skin_vac = A_alpha_vac + A_beta_vac
                
                sigma_sv = self._calc_sigma((phases[2],), T, v_mp[:,2], x_mp[:,2])
                G_surface += A_skin_vac * sigma_sv
                
            else:
                # 2. Alpha-Vacuum & Beta-Vacuum Interfaces
                sigma_av = self._calc_sigma((phases[0],), T, v_mp[:,0], x_mp[:,0])
                G_surface += A_alpha_outer * sigma_av
                
                sigma_bv = self._calc_sigma((phases[1],), T, v_mp[:,1], x_mp[:,1])
                G_surface += A_beta_outer * sigma_bv

        elif geometry_type == "Core_Shell":
            r_core, r_inner_total = r_vals
            A_core = 4 * np.pi * r_core**2
            A_inner_total = 4 * np.pi * r_inner_total**2
            
            # 1. Core-Shell Interface (Alpha-Beta)
            sigma_ab = self._calc_sigma((phases[0], phases[1]), T, v_mp[:,0], x_mp[:,0], v_mp[:,1], x_mp[:,1])
            G_surface += A_core * sigma_ab
            
            if has_skin:
                # 2. Beta-Skin Interface
                sigma_bs = self._calc_sigma((phases[1], phases[2]), T, v_mp[:,1], x_mp[:,1], v_mp[:,2], x_mp[:,2])
                G_surface += A_inner_total * sigma_bs
                
                # 3. Skin-Vacuum Interface
                V_total = np.sum(n_mp * v_mp)
                r_total = self.calc_r_from_V(V_total)
                A_skin_vac = 4 * np.pi * r_total**2
                
                sigma_sv = self._calc_sigma((phases[2],), T, v_mp[:,2], x_mp[:,2])
                G_surface += A_skin_vac * sigma_sv
                
            else:
                # 2. Beta-Vacuum Interface
                sigma_bv = self._calc_sigma((phases[1],), T, v_mp[:,1], x_mp[:,1])
                G_surface += A_inner_total * sigma_bv
        
        return G_surface
        
    def calc_G_ideal(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            T: float,
            phases: Tuple[str, ...],
    ) -> float:
        """
        Calculates the ideal Gibbs free energy for the system.
        n_mp: Mole matrix (rows: materials, columns: phases)
        x_mp: Mole fraction matrix (rows: materials, columns: phases)
        T: Temperature
        Returns the ideal Gibbs free energy.
        """
        R = 8.31446261815324  # J/(mol·K)
        G_ideal = 0.0
        num_phases = n_mp.shape[1]
        g_mp = self._get_g_mp(T, phases)

        for phase_idx in range(num_phases):
            n_phase = np.sum(n_mp[:, phase_idx])
            g_ideal_phase = x_mp[:,phase_idx] * g_mp[:,phase_idx] + R * T * x_mp[:,phase_idx] * np.log(x_mp[:,phase_idx] + 1e-20)
            G_ideal += n_phase * np.sum(g_ideal_phase)
        return G_ideal
    
    def _get_g_mp(self, T, phases):
        """
        Calculates the Gibbs free energy matrix for materials in the given phases.
        Rows correspond to materials (A, B), columns to the provided phases.
        """
        mat_names = self.system_data.config.materials
        g_mp_rows = []
        for mat_name in mat_names:
            row = []
            material = self.system_data.get_material(mat_name)
            for p in phases:
                # Use .get() for safe dictionary access to avoid KeyErrors
                phase_data = material.phases.get(p)
                row.append(phase_data.g0(T))
            g_mp_rows.append(row)

        return np.array(g_mp_rows)

    @staticmethod
    def calc_r_from_V(V):
        """
        Calculates the radius from volume for a sphere.
        V: Volume
        Returns radius.
        """
        return ((3 * V) / (4 * np.pi)) ** (1 / 3)

    def _get_v_mp(
            self,
            T: float,
            phases: Tuple[str, ...],
    ) -> np.ndarray:
        """
        Calculates the molar volume matrix for materials in the given phases.
        Rows correspond to materials (A, B), columns to the provided phases.
        """
        mat_names = self.system_data.config.materials
        v_mp_rows = []
        for mat_name in mat_names:
            row = []
            material = self.system_data.get_material(mat_name)
            for p in phases:
                # Use .get() for safe dictionary access to avoid KeyErrors
                phase_data = material.phases.get(p)
                
                # Check if phase_data and its 'v' attribute exist and are callable
                if phase_data and callable(phase_data.v):
                    row.append(phase_data.v(T))
                else:
                    # Raise a specific error if data is missing
                    raise ValueError(f"Molar volume function 'v' is missing or not callable for material '{mat_name}' in phase '{p}'.")
            v_mp_rows.append(row)

        return np.array(v_mp_rows)

    def _get_sigma_value(
            self,
            T: float,
            matA: str,
            phaseA: str,
            matB: str = "Vacuum",
            phaseB: str = None,
    ) -> float:
        sigma_data = self.system_data.surface_energy.AaBb[matA][phaseA]
        return sigma_data[matB][phaseB](T) if phaseB else sigma_data["Vacuum"](T)
    
    def _calc_sigma(
            self,
            curr_phases: Tuple[str, ...],
            T: float,
            v_gamma: np.ndarray,
            x_gamma: np.ndarray,
            v_delta: np.ndarray = None,
            x_delta: np.ndarray = None,
    ) -> float:
        """Calculates the interfacial energy between phase gamma and phase delta (or vacuum)."""
        phase_gamma_name = curr_phases[0]
        phase_delta_name = curr_phases[1] if len(curr_phases) > 1 else None
        
        v23x_gamma = v_gamma**(2/3) * x_gamma
        denom_gamma = np.sum(v23x_gamma)

        mats = self.system_data.config.materials
        total_sigma = 0.0

        # Case 1: Interface with another phase
        if v_delta is not None and x_delta is not None:
            v23x_delta = v_delta**(2/3) * x_delta
            denom_delta = np.sum(v23x_delta)
            
            for i, gamma_mat in enumerate(mats):
                for j, delta_mat in enumerate(mats):
                    sigma_val = self._get_sigma_value(T, gamma_mat, phase_gamma_name, delta_mat, phase_delta_name)
                    total_sigma += sigma_val * v23x_gamma[i] * v23x_delta[j]
            
            return total_sigma / (denom_gamma * denom_delta)
        
        # Case 2: Interface with vacuum
        else:
            for i, gamma_mat in enumerate(mats):
                sigma_val = self._get_sigma_value(T, gamma_mat, phase_gamma_name)
                total_sigma += sigma_val * v23x_gamma[i]
            
            return total_sigma / denom_gamma


    def _calc_mole_splits_and_geo(
            self,
            A_ratio_alpha: float, 
            B_ratio_alpha: float,
            T: float, 
            n_total: float, 
            xB_total: float, 
            phases: Tuple[str, ...],
            geometry_type: str,
            has_skin: bool = False,
            xB_skin: float = 1.0,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        def _calc_generic_split(n_A_alpha, n_B_alpha, n_A_total, n_B_total) -> Tuple[np.ndarray, np.ndarray]:
            n_A_beta = n_A_total - n_A_alpha
            n_B_beta = n_B_total - n_B_alpha
            n_mp = np.array([
                [n_A_alpha, n_A_beta],
                [n_B_alpha, n_B_beta],
            ])

            n_alpha = n_A_alpha + n_B_alpha
            n_beta = n_A_beta + n_B_beta
            xA_alpha = n_A_alpha / n_alpha
            xB_alpha = n_B_alpha / n_alpha
            xA_beta = n_A_beta / n_beta
            xB_beta = n_B_beta / n_beta

            x_mp = np.array([
                [xA_alpha, xA_beta],
                [xB_alpha, xB_beta],
            ])

            return n_mp, x_mp

        n_A_total = n_total * (1 - xB_total)
        n_B_total = n_total * xB_total
        
        if has_skin:
            v_mp = self._get_v_mp(T, phases)
            n_A_alpha = A_ratio_alpha * n_A_total
            n_B_alpha = B_ratio_alpha * n_B_total
            n_no_skin_mp, x_no_skin_mp = _calc_generic_split(n_A_alpha, n_B_alpha, n_A_total, n_B_total)
            V_no_skin = np.sum(n_no_skin_mp * v_mp[:,:2])
            r_no_skin = self.calc_r_from_V(V_no_skin)
            mats = self.system_data.config.materials
            weighted_skin_thickness = 2 * (1 - xB_skin) * self.system_data.material_data[mats[0]].atomic_radius + 2 * xB_skin * self.system_data.material_data[mats[1]].atomic_radius
            weighted_skin_molar_volume = (1 - xB_skin) * v_mp[0,2] + xB_skin * v_mp[1,2]
            V_skin_prev_guess = (4/3) * np.pi * (r_no_skin ** 3 - (r_no_skin - weighted_skin_thickness) ** 3)
            n_skin_prev_guess = V_skin_prev_guess / weighted_skin_molar_volume
            
            num_of_tries = 0
            was_successful = False
            while num_of_tries < 100:
                num_of_tries += 1
                n_A_skin = n_skin_prev_guess * (1 - xB_skin)
                n_B_skin = n_skin_prev_guess * xB_skin
                n_A_no_skin = n_A_total - n_A_skin
                n_B_no_skin = n_B_total - n_B_skin

                if n_A_no_skin < 0 or n_B_no_skin < 0:
                    # Not enough material to form the requested skin, this path is invalid.
                    break

                n_A_alpha = A_ratio_alpha * n_A_no_skin
                n_B_alpha = B_ratio_alpha * n_B_no_skin 
                n_mp_no_skin, x_mp_no_skin = _calc_generic_split(n_A_alpha, n_B_alpha, n_A_no_skin, n_B_no_skin)

                match geometry_type:
                    case "Janus":
                        r_vals = self.calc_Janus_geometry_for_known_nx(
                            n_mp_no_skin,
                            x_mp_no_skin,
                            phases,
                            T,
                            xB_skin,
                        )
                        h_a, h_b, a_i = r_vals
                        r_a = (h_a**2 + a_i**2) / (2 * h_a)
                        r_b = (h_b**2 + a_i**2) / (2 * h_b)
                        V_no_skin = np.sum(n_mp_no_skin * v_mp[:,:2])
                        r_a_with_skin = r_a + weighted_skin_thickness
                        r_b_with_skin = r_b + weighted_skin_thickness
                        theta_a = np.arcsin(np.clip(a_i / r_a, -1.0, 1.0))
                        theta_b = np.arcsin(np.clip(a_i / r_b, -1.0, 1.0))
                        V_a_with_skin = (1/3) * np.pi * (r_a_with_skin**3) * (2 + np.cos(theta_a)) * (1 - np.cos(theta_a))**2
                        V_b_with_skin = (1/3) * np.pi * (r_b_with_skin**3) * (2 + np.cos(theta_b)) * (1 - np.cos(theta_b))**2
                        V_skin = V_a_with_skin + V_b_with_skin - V_no_skin
                        n_skin_curr_guess = V_skin / weighted_skin_molar_volume

                    case "Core_Shell":
                        r_vals = self.calc_core_shell_geometry_for_known_nx(
                            n_mp_no_skin,
                            x_mp_no_skin,
                            phases,
                            T,
                        )
                        V_no_skin = np.sum(n_mp_no_skin * v_mp[:,:2])
                        r_with_skin = r_vals[1] + weighted_skin_thickness
                        V_with_skin = (4/3) * np.pi * (r_with_skin ** 3)
                        V_skin = V_with_skin - V_no_skin
                        n_skin_curr_guess = V_skin / weighted_skin_molar_volume
                    
                if abs(n_skin_curr_guess - n_skin_prev_guess) < n_total * 1e-6:
                    was_successful = True
                    break
                else:
                    n_skin_prev_guess = n_skin_curr_guess*0.5 + n_skin_prev_guess*0.5

            if not was_successful:
                raise ValueError("Failed to converge skin mole calculation or insufficient moles for skin.")
            else:
                n_mp = np.array([
                    [n_A_alpha, n_A_no_skin - n_A_alpha, n_A_skin],
                    [n_B_alpha, n_B_no_skin - n_B_alpha, n_B_skin],
                ])
                n_alpha = np.sum(n_mp[:,0])
                n_beta = np.sum(n_mp[:,1])
                n_skin = np.sum(n_mp[:,2])
                xA_alpha = n_A_alpha / n_alpha
                xB_alpha = n_B_alpha / n_alpha
                xA_beta = (n_A_no_skin - n_A_alpha) / n_beta
                xB_beta = (n_B_no_skin - n_B_alpha) / n_beta
                xA_skin = n_A_skin / n_skin
                xB_skin = n_B_skin / n_skin
                x_mp = np.array([
                    [xA_alpha, xA_beta, xA_skin],
                    [xB_alpha, xB_beta, xB_skin],
                ])
            
        else:
            n_A_alpha = A_ratio_alpha * n_A_total
            n_B_alpha = B_ratio_alpha * n_B_total
            n_mp, x_mp = _calc_generic_split(n_A_alpha, n_B_alpha, n_A_total, n_B_total)

            match geometry_type:
                case "Janus":
                    r_vals = self.calc_Janus_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T,
                    )

                case "Core_Shell":
                    r_vals = self.calc_core_shell_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T,
                    )
        return n_mp, x_mp, r_vals  # Return the calculated mole splits and compositions
    
    def calc_Janus_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T: float,
            xB_skin: float = None,
    ) -> np.ndarray:
        """
        Calculates the geometry parameters for Janus particles.
        """
        v_mp = self._get_v_mp(T, phases)
        outer_phase_name = next(iter(phases[2:]), None)
        v_outer_phase = v_mp[:, 2] if outer_phase_name else None
        if xB_skin is not None:
            x_outer_phase = np.array([1 - xB_skin, xB_skin])
        else:
            x_outer_phase = None

        sigma_outer = [
            self._calc_sigma(
                (phases[i], outer_phase_name),
                T,
                v_mp[:, i],
                x_mp[:, i],
                v_delta=v_outer_phase,
                x_delta=x_outer_phase,
            )
            for i in range(2)
        ]
        sigma_alpha_out, sigma_beta_out = sigma_outer

        sigma_interface = self._calc_sigma(
            (phases[0], phases[1]),
            T,
            v_mp[:,0],
            x_mp[:,0],
            v_delta=v_mp[:,1],
            x_delta=x_mp[:,1],
        )
        nv_mp = n_mp * v_mp[:, :2]
        def helper_calc_Janus_geo(sigma_interface, sigma_alpha_out, sigma_beta_out):
            V_total = np.sum(nv_mp)
            r_total = self.calc_r_from_V(V_total)
            h_initial_guesses = np.array([r_total, r_total])
            def helper_Janus_geo_ri_calc(h_alpha_beta):
                h_a, h_b = h_alpha_beta
                h2_a = h_a**2
                h2_b = h_b**2
                s_i = sigma_interface
                s_a = sigma_alpha_out
                s_b = sigma_beta_out
                ri2_calc_coeff = [
                            s_a + s_b + s_i,
                            (h2_a + h2_b) * s_i - (s_a - s_b) * (h2_a - h2_b) * s_i,
                            h2_a * h2_b * (s_i - s_a - s_b)
                        ]
                ri2_roots = np.roots(ri2_calc_coeff)
                ri2_real_positive = ri2_roots[np.isreal(ri2_roots) & (ri2_roots > 0)].real
                if len(ri2_real_positive) == 0:
                    return [1e5, 1e5]
                ri2_real_positive = np.min(ri2_real_positive)
                r_i = ri2_real_positive**0.5
                eq1 = ((1/6) * np.pi * h_a) * (3 * (r_i ** 2) + h_a ** 2) - np.sum(nv_mp[:,0])
                eq2 = ((1/6) * np.pi * h_b) * (3 * (r_i ** 2) + h_b ** 2) - np.sum(nv_mp[:,1])

                return [eq1, eq2]
            sol = optimize.root(helper_Janus_geo_ri_calc, h_initial_guesses, method='hybr')
            
            if not sol.success:
                raise ValueError("Janus geometry solver failed to converge.")

            term_inside_sqrt = (((6 * np.sum(nv_mp[:,0])) / (np.pi * sol.x[0])) - sol.x[0] ** 2) / 3
            if term_inside_sqrt < 0:
                raise ValueError("Unphysical Janus Geometry: Negative interface radius squared.")
            
            a_i_result = np.sqrt(term_inside_sqrt)
            h_a_res, h_b_res = sol.x
            r_vals = np.array([h_a_res, h_b_res, a_i_result]) ### r_vals def - Janus no Skin
            return r_vals
        r_vals = helper_calc_Janus_geo(sigma_interface, sigma_alpha_out, sigma_beta_out)
        return r_vals

    def calc_core_shell_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T: float,
    ) -> np.ndarray:
        """
        Calculates the geometry parameters for Core-Shell particles.
        """
        v_mp = self._get_v_mp(T, phases)
        V_alpha = np.sum(n_mp[:,0] * v_mp[:,0])
        V_total = np.sum(n_mp[:,:2] * v_mp[:,:2])
        r_alpha = self.calc_r_from_V(V_alpha)
        r_total = self.calc_r_from_V(V_total)
        r_vals = np.array([r_alpha, r_total]) ### r_vals def - Core-Shell no Skin
        return r_vals