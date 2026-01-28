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
        
        if has_skin:
            phases = primary_phases + ("Liquid",)
        else:
            phases = primary_phases
        

        n_mp, x_mp, g_mp = self._calc_mole_splits_and_geo(
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
    
        return 0.0  # Placeholder for actual Gibbs energy calculation logic
    
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

        v_mp = np.array([
            [self.system_data.get_material(mat_name).phases[p].v(T) for p in phases]
            for mat_name in mat_names
        ])

        return v_mp

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
            v_delta: np.ndarray = 1.0,
            x_delta: np.ndarray = 1.0,
    ) -> float:

        v23x_gamma = v_gamma**(2/3) * x_gamma
        v23x_delta = v_delta**(2/3) * x_delta
        denom = np.sum(v23x_gamma) * np.sum(v23x_delta)

        total_sigma = 0.0
        mats = self.system_data.config.materials
        for i, gamma_mat in enumerate(mats):
            for j, delta_mat in enumerate(mats):
                curr_sigma = self._get_sigma_value(T, gamma_mat, curr_phases[0], delta_mat, next(iter(curr_phases[1:]), None))
                v23x_delta_slice = v23x_delta[j] if j < len(v23x_delta) else 1.0
                total_sigma += curr_sigma * v23x_gamma[i] * v23x_delta_slice / denom
        return total_sigma


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

        if has_skin:

            pass # Placeholder logic once skin exists
        else:
            n_A_total = n_total * (1 - xB_total)
            n_B_total = n_total * xB_total
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
                    v_mp = self._get_v_mp(T, phases)
                    V_alpha = np.sum(n_mp[:,0] * v_mp[:,0])
                    V_total = np.sum(n_mp * v_mp)
                    r_alpha = self.calc_r_from_V(V_alpha)
                    r_total = self.calc_r_from_V(V_total)
                    r_vals = np.array([r_alpha, r_total]) ### r_vals def - Core-Shell no Skin

        return n_mp, x_mp, r_vals  # Return the calculated mole splits and compositions
    
    def calc_Janus_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T: float,
    ) -> np.ndarray:
        """
        Calculates the geometry parameters for Janus particles.
        """
        v_mp = self._get_v_mp(T, phases)
        outer_phase = next(iter(phases[2:]), None)
        v_outer_phase = v_mp[:,2] if outer_phase else 1.0
        x_outer_phase = x_mp[:,2] if outer_phase else 1.0
        sigma_outer = [
            self._calc_sigma(
            phases,
            T,
            v_mp[:,i],
            x_mp[:,i],
            v_outer_phase,
            x_outer_phase,
        ) 
        for i in range(2)
        ]
        sigma_alpha_out, sigma_beta_out = sigma_outer
        sigma_interface = self._calc_sigma(
            phases,
            T,
            v_mp[:,0],
            x_mp[:,0],
            v_mp[:,1],
            x_mp[:,1],
        )
        nv_mp = n_mp * v_mp
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
            term_inside_sqrt = (((6 * np.sum(nv_mp[:,0])) / (np.pi * sol.x[0])) - sol.x[0] ** 2) / 3
            a_i_result = (term_inside_sqrt ** (1/2))
            r_vals = np.array([sol.x[0], sol.x[1], a_i_result]) ### r_vals def - Janus no Skin
            return r_vals
        r_vals = helper_calc_Janus_geo(sigma_interface, sigma_alpha_out, sigma_beta_out)
        return r_vals
