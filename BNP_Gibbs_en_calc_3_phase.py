import numpy as np
import inspect
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from configurations_3_phase import ThreePhaseConfiguration
import pure_material_properties
import interaction_properties
from scipy import optimize

class skin_class:
    def __init__(self, xB):
        if xB is not None and 0 <= xB <= 1:
            self.xB = xB
            self.exists = True
        else:
            self.exists = False

@dataclass
class TemperatureDependentVars:
    T: float
    v_mp: np.ndarray
    g_mp: np.ndarray
    L_ip: Dict[str, np.ndarray]
    st_mp: np.ndarray

class GibbsEnergyCalculator3Phase:
    """
    Calculates the total Gibbs free energy for a 3-phase system.
    Designed to accept a generic list of optimization variables.
    """
    def __init__(self, config: ThreePhaseConfiguration):
        self.config = config
        self.material_data = {}

        for _, obj in inspect.getmembers(pure_material_properties):
            if inspect.isclass(obj) and issubclass(obj, pure_material_properties.BaseMaterial) and obj is not pure_material_properties.BaseMaterial:
                instance = obj()
                if instance.name in self.config.materials:
                    self.material_data[instance.name] = instance

        self.interaction_data = None
        materials_frozenset = frozenset(self.config.materials)
        for _, obj in inspect.getmembers(interaction_properties):
            if inspect.isclass(obj) and issubclass(obj, interaction_properties.BaseInteraction) and obj is not interaction_properties.BaseInteraction:
                instance = obj()
                if hasattr(instance, 'names') and frozenset(instance.names) == materials_frozenset:
                    self.interaction_data = instance

    def _get_g_mp(self, T, phases):
        """
        Calculates the Gibbs free energy matrix for materials in the given phases.
        Rows correspond to materials (A, B), columns to the provided phases.
        """
        mat_names = self.config.materials
        g_mp_rows = []
        for mat_name in mat_names:
            row = []
            material = self.material_data[mat_name]
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
        mat_names = self.config.materials
        v_mp_rows = []
        for mat_name in mat_names:
            row = []
            material = self.material_data[mat_name]
            for p in phases:
                phase_data = material.phases.get(p)
                row.append(phase_data.v(T))
            v_mp_rows.append(row)

        return np.array(v_mp_rows)
    
    def _get_T_dependent_vars(self, T: float, phases: Tuple[str, ...]) -> TemperatureDependentVars:
        v_mp = self._get_v_mp(T, phases)
        g_mp = self._get_g_mp(T, phases)
        L_ip = {p: self.interaction_data.phases[p].get_Li_per_T(T) for p in phases} # type: ignore
        
        st_mp = np.array([
            [
                self.material_data[m].phases[p].surface_tension(T)
                for p in phases
            ]
            for m in self.config.materials
        ])
        return TemperatureDependentVars(T, v_mp, g_mp, L_ip, st_mp)

    def calculate_total_energy(self, 
                               A_ratio_alpha: float, 
                               B_ratio_alpha: float,
                               T: float, 
                               n_total: float, 
                               xB_total: float, 
                               primary_phases: Tuple[str, str],
                               geometry_type: str,
                               skin_val: float = None, # type: ignore
                            ) -> float:
        
        def _update_phases_based_on_skin(primary_phases, skin_val):
            skin = skin_class(skin_val)
            if skin.exists:
                phases = primary_phases + ("Liquid",)
            else:
                phases = primary_phases
            return phases, skin

        try:
            phases, skin = _update_phases_based_on_skin(primary_phases, skin_val)
            
            vars = self._get_T_dependent_vars(T, phases)

            n_mp, x_mp, r_vals = self._calc_mole_splits_and_geo(
                A_ratio_alpha, 
                B_ratio_alpha,
                n_total, 
                xB_total, 
                phases,
                geometry_type,
                skin,
                T,
                vars
            )
    #         x_mp = np.clip(x_mp, 1e-20, 1.0 - 1e-20)  # Prevent log(0) issues
    #         G_ideal = self.calc_G_ideal(n_mp, x_mp, T, phases)
    #         G_excess = self.calc_G_excess(n_mp, x_mp, T, phases)
    #         G_surface = self.calc_G_surface(n_mp, x_mp, r_vals, T, phases, geometry_type, has_skin)

    #         return G_ideal + G_excess + G_surface
            print("n_mp:", n_mp)
            print("x_mp:", x_mp)
            print("r_vals:", r_vals)
            
            return 0.0

        except ValueError: # Anytime something returns "no solution"
            return 1.0
    
    def _calc_mole_splits_and_geo(
            self,
            A_ratio_alpha: float, 
            B_ratio_alpha: float,
            n_total: float, 
            xB_total: float, 
            phases: Tuple[str, ...],
            geometry_type: str,
            skin: skin_class,
            T: float,
            vars: TemperatureDependentVars,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # This function will calculate the mole splits and geometry parameters based on the provided inputs.
        # It will handle both cases of having a skin or not.

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
        
        if skin.exists:

            # Guess initial mole amounts based on ratios

            match geometry_type:
                case "Janus":
                    pass
                case "Core Shell": 
                    
                    # Begin loop
                    # Calculate geo based on intial estimate
                    # See if geo is good, if not, update guess and repeat until convergence or max iterations
                    # Calculate total r (include skin thickness))
                    
                    pass
            
            pass
        else:

            n_A_alpha = A_ratio_alpha * n_A_total
            n_B_alpha = B_ratio_alpha * n_B_total
            n_mp, x_mp = _calc_generic_split(n_A_alpha, n_B_alpha, n_A_total, n_B_total)
            
            match geometry_type:
                case "Janus":
                    r_vals = self._calc_Janus_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T,
                        skin,
                        vars,
                    )
                case "Core Shell":
                    r_vals = self._calc_core_shell_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        vars,
                    )
        return n_mp, x_mp, r_vals
        
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
            g_ideal_phase = x_mp[:,phase_idx] * g_mp[:,phase_idx] + R * T * x_mp[:,phase_idx] * np.log(x_mp[:,phase_idx])
            G_ideal += n_phase * np.sum(g_ideal_phase)
        return G_ideal
    

    def _calc_Janus_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T: float,
            skin: skin_class,
            vals: TemperatureDependentVars,
    ) -> np.ndarray:
        
        def _calculate_spheric_Janus_geo():
            V_alpha = np.sum(n_mp[:,0] * vals.v_mp[:,0])
            V_beta = np.sum(n_mp[:,1] * vals.v_mp[:,1])
            V_ratio = V_alpha / V_beta
            cos_theta_alpha = 0.45
            tol_check = 1.0
            num_iterations = 0
            max_iterations = 100
            while tol_check > 1e-6 and num_iterations < max_iterations:
                num_iterations += 1
                cos_theta_curr = V_ratio *((2 - cos_theta_alpha) * (1+cos_theta_alpha) ** 2 ) / ((1 - cos_theta_alpha) ** 2)  - 2
                tol_check = abs(cos_theta_curr - cos_theta_alpha)
                cos_theta_alpha = 0.9 * cos_theta_curr + 0.1 * cos_theta_alpha
            
            if num_iterations == max_iterations:
                raise ValueError("Spheric Janus geometry solver failed to converge.")
            
            r_alpha = ((3 * V_alpha) / (np.pi * (2 - cos_theta_alpha) * (1 + cos_theta_alpha) ** 2)) ** (1/3)
            r_beta = ((3 * V_beta) / (np.pi * (2 + cos_theta_alpha) * (1 - cos_theta_alpha) ** 2)) ** (1/3)
            return np.array([r_alpha, r_beta, cos_theta_alpha])
        def _calculate_surface_tension_val(xB_alpha, xB_beta, phase_alpha, phase_beta):
            def _calculate_G_excess_for_surface_tension_same_phase(xB, k, phase, is_dG_A):
                if not is_dG_A:
                    xB = 1 - xB
                L_params = vals.L_ip[phase]
                return k*(xB**2)*(L_params[0]+L_params[1]*(3-4*xB))
            def _calculate_omega_for_surface_tension(vars, phase_num, mat_num, f):
                return f*(vars.v_mp[mat_num, phase_num]**(2/3))*(6.02214e23)**(1/3)

            
            def _calculate_surface_tension_alloy_to_vacuum(xB, phase):
                k = 0.804 if phase == "Liquid" else 0.75
                f = 1.0 if phase == "Liquid" else 1.09 
                phase_num = phases.index(phase)
                mat_num = 
                omega_A = _calculate_omega_for_surface_tension
                

                pass
            def _calculate_surface_tension_solid_to_solid(T, vals, xB_alpha, xB_beta, phase_alpha, phase_beta):
                pass
            def _calculate_surface_tension_liquid_to_liquid(T, vals, xB_alpha, xB_beta, phase_alpha="Liquid", phase_beta="Liquid"):
                pass
            def _calculate_surface_tension_solid_to_liquid(T, vals, xB_solid, xB_liquid, phase_solid, phase_liquid="Liquid"):
                pass
            
            
            match (phase_alpha, phase_beta):
                case ("Liquid", "Liquid"):
                    st = _calculate_surface_tension_liquid_to_liquid(T, vals, xB_alpha, xB_beta)
                case ("FCC", None) | ("Liquid", None):
                    st = _calculate_surface_tension_alloy_to_vacuum(T, vals, xB_alpha, phase_alpha)
                case (p_a, p_b) if p_b is not None and "Liquid" in (p_a, p_b) and p_a != p_b:
                    st = _calculate_surface_tension_solid_to_liquid(T, vals, xB_alpha if p_a != "Liquid" else xB_beta, xB_beta if p_b != "Liquid" else xB_alpha, phase_alpha if p_a != "Liquid" else phase_beta, phase_beta if p_b == "Liquid" else phase_alpha)
                case (p_a, p_b) if p_b is not None and "Liquid" not in (p_a, p_b):
                    st = _calculate_surface_tension_solid_to_solid(T, vals, xB_alpha, xB_beta, phase_alpha, phase_beta)
            return st
            
        spheric_r_vals = _calculate_spheric_Janus_geo(n_mp, vals)
        st_a = _calculate_surface_tension_val(x_mp[1,0], skin.xB if skin.exists else None, phases[0], phases[2] if skin.exists else None, T, vals)
        st_b = _calculate_surface_tension_val(x_mp[1,1], skin.xB if skin.exists else None, phases[1], phases[2] if skin.exists else None, T, vals)
        st_i = _calculate_surface_tension_val(x_mp[1,0], x_mp[1,1], phases[0], phases[1], T, vals)


        # Calculate spheric geo (initial guess + useful later if fails)
        # Calculate the surface tensions based on Kaptay and db vals - NOTE THAT IF SKIN EXISTS, VALUES ARE FOR OUTERPHASE
        # Solve h_alpha h_beta based (r_i is known based on them)
        # If fails, return spheric solution
        
        #region Old code

        # v_mp = vars.v_mp
        # outer_phase_name = next(iter(phases[2:]), None)
        # v_outer_phase = v_mp[:, 2] if outer_phase_name else None
        # if xB_skin is not None:
        #     x_outer_phase = np.array([1 - xB_skin, xB_skin])
        # else:
        #     x_outer_phase = None

        # sigma_outer = [
        #     self._calc_sigma(
        #         (phases[i], outer_phase_name),
        #         T,
        #         v_mp[:, i],
        #         x_mp[:, i],
        #         v_delta=v_outer_phase,
        #         x_delta=x_outer_phase,
        #     )
        #     for i in range(2)
        # ]
        # sigma_alpha_out, sigma_beta_out = sigma_outer

        # sigma_interface = self._calc_sigma(
        #     (phases[0], phases[1]),
        #     T,
        #     v_mp[:,0],
        #     x_mp[:,0],
        #     v_delta=v_mp[:,1],
        #     x_delta=x_mp[:,1],
        # )
        # nv_mp = n_mp * v_mp[:, :2]
        # def helper_calc_Janus_geo(sigma_interface, sigma_alpha_out, sigma_beta_out):
        #     V_total = np.sum(nv_mp)
        #     r_total = self.calc_r_from_V(V_total)
        #     h_initial_guesses = np.array([r_total, r_total])
        #     def helper_Janus_geo_ri_calc(h_alpha_beta):
        #         h_a, h_b = h_alpha_beta
        #         h2_a = h_a**2
        #         h2_b = h_b**2
        #         s_i = sigma_interface
        #         s_a = sigma_alpha_out
        #         s_b = sigma_beta_out
        #         ri2_calc_coeff = [
        #                     s_a + s_b + s_i,
        #                     (h2_a + h2_b) * s_i - (s_a - s_b) * (h2_a - h2_b) * s_i,
        #                     h2_a * h2_b * (s_i - s_a - s_b)
        #                 ]
        #         ri2_roots = np.roots(ri2_calc_coeff)
        #         ri2_real_positive = ri2_roots[np.isreal(ri2_roots) & (ri2_roots > 0)].real
        #         if len(ri2_real_positive) == 0:
        #             return [1e5, 1e5]
        #         ri2_real_positive = np.min(ri2_real_positive)
        #         r_i = ri2_real_positive**0.5
        #         eq1 = ((1/6) * np.pi * h_a) * (3 * (r_i ** 2) + h_a ** 2) - np.sum(nv_mp[:,0])
        #         eq2 = ((1/6) * np.pi * h_b) * (3 * (r_i ** 2) + h_b ** 2) - np.sum(nv_mp[:,1])

        #         return [eq1, eq2]
        #     sol = optimize.root(helper_Janus_geo_ri_calc, h_initial_guesses, method='hybr')
            
        #     if not sol.success:
        #         raise ValueError("Janus geometry solver failed to converge.")

        #     term_inside_sqrt = (((6 * np.sum(nv_mp[:,0])) / (np.pi * sol.x[0])) - sol.x[0] ** 2) / 3
        #     if term_inside_sqrt < 0:
        #         raise ValueError("Unphysical Janus Geometry: Negative interface radius squared.")
            
        #     a_i_result = np.sqrt(term_inside_sqrt)
        #     h_a_res, h_b_res = sol.x
        #     r_vals = np.array([h_a_res, h_b_res, a_i_result]) ### r_vals def - Janus no Skin
        #     return r_vals
        # r_vals = helper_calc_Janus_geo(sigma_interface, sigma_alpha_out, sigma_beta_out)
        #endregion
        return r_vals

    def _calc_core_shell_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            vars: TemperatureDependentVars,
    ) -> np.ndarray:
        """
        Calculates the geometry parameters for Core-Shell particles.
        """
        v_mp = vars.v_mp
        V_alpha = np.sum(n_mp[:,0] * v_mp[:,0])
        V_core_and_shell = np.sum(n_mp[:,:2] * v_mp[:,:2])
        r_core = self.calc_r_from_V(V_alpha)
        r_core_and_shell = self.calc_r_from_V(V_core_and_shell)
        r_vals = np.array([r_core, r_core_and_shell]) ### r_vals def - Core-Shell no Skin
        return r_vals