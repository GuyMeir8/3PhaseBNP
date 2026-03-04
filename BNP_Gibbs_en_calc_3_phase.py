import numpy as np
import inspect
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from configurations_3_phase import ThreePhaseConfiguration
import pure_material_properties
import interaction_properties
from scipy import optimize
from math import log, exp, sqrt

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
            T_dependent_parameters = self._get_T_dependent_vars(T, phases)
            n_mp, x_mp, r_vals = self._calc_mole_splits_and_geo(
                A_ratio_alpha, 
                B_ratio_alpha,
                n_total, 
                xB_total, 
                phases,
                geometry_type,
                skin,
                T,
                T_dependent_parameters
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
            T_dependent_parameters: TemperatureDependentVars,
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
                        T_dependent_parameters,
                    )
                case "Core Shell":
                    r_vals = self._calc_core_shell_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T_dependent_parameters,
                    )
        return n_mp, x_mp, r_vals
        
    def _calc_G_ideal(
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
    
    def _generic_iterative_loop_one_variable(
            self,
            initial_guess: float,
            update_function: Callable[[float], float],
            tol: float = 1e-6,
            max_iterations: int = 100,
    ) -> Tuple[float, int]:

        var_current = initial_guess
        for iteration in range(max_iterations):
            var_next = update_function(var_current)
            if abs(var_next - var_current) < tol:
                return (var_next, iteration + 1)
            var_current = var_next*0.9 + var_current*0.1 
        return (var_next, max_iterations)

    def _calc_Janus_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T: float,
            skin: skin_class,
            T_dependent_parameters: TemperatureDependentVars,
    ) -> np.ndarray:
        
        def _calculate_spheric_Janus_geo():
            V_alpha = np.sum(n_mp[:,0] * T_dependent_parameters.v_mp[:,0])
            V_beta = np.sum(n_mp[:,1] * T_dependent_parameters.v_mp[:,1])
            V_ratio = V_alpha / V_beta
            def cos_theta_update(cos_theta_alpha):
                return V_ratio *((2 - cos_theta_alpha) * (1+cos_theta_alpha) ** 2 ) / ((1 - cos_theta_alpha) ** 2)  - 2
            
            max_iterations = 100
            cos_theta_alpha, num_iterations = self._generic_iterative_loop_one_variable(
                initial_guess=0.45,
                update_function=cos_theta_update,
                tol=1e-6,
                max_iterations=max_iterations
            )
            
            if num_iterations == max_iterations:
                raise ValueError("Spheric Janus geometry solver failed to converge.")
            
            r_alpha = ((3 * V_alpha) / (np.pi * (2 - cos_theta_alpha) * (1 + cos_theta_alpha) ** 2)) ** (1/3)
            r_beta = ((3 * V_beta) / (np.pi * (2 + cos_theta_alpha) * (1 - cos_theta_alpha) ** 2)) ** (1/3)
            return np.array([r_alpha, r_beta, cos_theta_alpha])
        
        def _calculate_surface_tension_val(xB_alpha, xB_beta, phase_alpha, phase_beta):
            def _calculate_G_excess_for_surface_tension_same_phase(xB, k, phase, is_dG_A):
                if not is_dG_A:
                    xB = 1 - xB
                L_params = T_dependent_parameters.L_ip[phase]
                return k*(xB**2)*(L_params[0]+L_params[1]*(3-4*xB))
            def _calculate_omega_for_surface_tension(v_i, f):
                return f*(v_i**(2/3))*(6.02214e23)**(1/3)
            def _calculate_G_excess_for_surface_tension_different_phases_interface(xB_i, k, phase_alpha, phase_beta, is_dG_A):
                if not is_dG_A:
                    xB_i = 1 - xB_i
                L_alpha_0 = T_dependent_parameters.L_ip[phase_alpha][0]
                L_beta_0 = T_dependent_parameters.L_ip[phase_beta][0]
                return k*(xB_i**2)*(0.5*(L_alpha_0 + L_beta_0))
            
            def _calculate_surface_tension_alloy_to_vacuum(xB, phase):
                k = 0.804 if phase == "Liquid" else 0.75
                f = 1.0 if phase == "Liquid" else 1.09 
                phase_num = phases.index(phase)
                v_A = T_dependent_parameters.v_mp[0, phase_num]
                v_B = T_dependent_parameters.v_mp[1, phase_num]
                omega_A = _calculate_omega_for_surface_tension(v_A, f)
                omega_B = _calculate_omega_for_surface_tension(v_B, f)
                dG_A_b = _calculate_G_excess_for_surface_tension_same_phase(xB, k, phase, is_dG_A=True)
                dG_B_b = _calculate_G_excess_for_surface_tension_same_phase(xB, k, phase, is_dG_A=False)
                R = 8.31446261815324 
                dst_0 = T_dependent_parameters.st_mp[0, phase_num] - T_dependent_parameters.st_mp[1, phase_num]
                def update_xB_i(xB_i_prev):
                    dG_A_i = _calculate_G_excess_for_surface_tension_same_phase(xB_i_prev, k, phase, is_dG_A=True)
                    dG_B_i = _calculate_G_excess_for_surface_tension_same_phase(xB_i_prev, k, phase, is_dG_A=False)
                    return xB * exp((dst_0 + R * T * log((1 - xB_i_prev) / (1 - xB)) / omega_A + (dG_A_i - dG_A_b) / omega_A - (dG_B_i - dG_B_b) / omega_B) * omega_B / (R * T))
                max_iterations = 100
                xB_i, num_iterations = self._generic_iterative_loop_one_variable(
                    initial_guess=xB,
                    update_function=update_xB_i,
                    tol=1e-6,
                    max_iterations=max_iterations)

                if num_iterations == max_iterations:
                    raise ValueError("Alloy to Vacuum surface tension solver failed to converge.")
                
                dG_A_i = _calculate_G_excess_for_surface_tension_same_phase(xB_i, k, phase, is_dG_A=True)
                st_vac_alloy = T_dependent_parameters.st_mp[0, phase_num] + ( R * T / omega_A) * log((1 - xB_i) / (1 - xB)) + (dG_A_i - dG_A_b) / omega_A
                return st_vac_alloy
            
            def _calculate_surface_tension_solid_to_solid(xB_alpha, xB_beta, phase_alpha, phase_beta):
                v_alpha = np.sum(T_dependent_parameters.v_mp[:, phases.index(phase_alpha)] * x_mp[:,phases.index(phase_alpha)])
                v_beta = np.sum(T_dependent_parameters.v_mp[:, phases.index(phase_beta)] * x_mp[:,phases.index(phase_beta)])
                z_V = v_alpha / v_beta - 1 if v_alpha > v_beta else v_beta / v_alpha - 1
                R = 8.31446261815324
                st_A_solid_solid = T_dependent_parameters.st_mp[0, phases.index(phase_alpha)] * (1/3)
                st_B_solid_solid = T_dependent_parameters.st_mp[1, phases.index(phase_beta)] * (1/3)
                def _calculate_incoherent_solid_solid():
                    k_incoherent = 0.917
                    f_incoherent = 1.045
                    omega_A_incoherent = _calculate_omega_for_surface_tension(T_dependent_parameters.st_mp[0, phases.index(phase_alpha)], f_incoherent)
                    omega_B_incoherent = _calculate_omega_for_surface_tension(T_dependent_parameters.st_mp[1, phases.index(phase_beta)], f_incoherent)
                    L_0_FCC = T_dependent_parameters.L_ip["FCC"][0]
                    dst_ss = st_A_solid_solid - st_B_solid_solid
                    def update_xB_i(xB_i_prev):
                        dS = R * T * (log((1 - xB_i_prev) / (1 - xB_alpha)) / omega_A_incoherent - log(xB_i_prev / xB_alpha) / omega_B_incoherent)
                        dEx = L_0_FCC * ((k_incoherent*xB_i_prev**2 - xB_alpha**2)/omega_A_incoherent - (1-xB_alpha**2)/omega_B_incoherent)
                        return 1 - sqrt((dst_ss + dS + dEx) / k_incoherent)
                    max_iterations = 100
                    xB_i, num_iterations = self._generic_iterative_loop_one_variable(
                        initial_guess=0.5*(xB_alpha + xB_beta),
                        update_function=update_xB_i,
                        tol=1e-6,
                        max_iterations=max_iterations)
                    if num_iterations == max_iterations:
                        raise ValueError("Solid to Solid surface tension solver failed to converge.")
                    return st_A_solid_solid + (R*T/omega_A_incoherent)*log((1-xB_i)/(1-xB_alpha))+(L_0_FCC/omega_A_incoherent)*((k_incoherent*xB_i**2 - xB_alpha**2))
                st_incoherent = _calculate_incoherent_solid_solid()
                if z_V >= 0.47:
                    return st_incoherent
                def _calculate_coherent_solid_solid():
                    k = 1.0
                    f = 1.09
                    omega_coherent = _calculate_omega_for_surface_tension(T_dependent_parameters.st_mp[0, phases.index(phase_alpha)], f)
                    L_0_alpha = T_dependent_parameters.L_ip[phase_alpha][0]
                    L_0_beta = T_dependent_parameters.L_ip[phase_beta][0]
                    L_ave = 0.5*(L_0_alpha + L_0_beta)
                    def update_xB_i(xB_i_prev):
                        dS = R * T * log((xB_i_prev*(1-xB_i_prev))/ (xB_alpha*(1-xB_alpha)))
                        dEx = L_ave*xB_i_prev**2 - L_0_alpha*(1 - 2*xB_alpha)
                        return 1 - sqrt((dS + dEx) / k) 
                    max_iterations = 100
                    xB_i, num_iterations = self._generic_iterative_loop_one_variable(
                        initial_guess=0.5*(xB_alpha + xB_beta),
                        update_function=update_xB_i,
                        tol=1e-6,
                        max_iterations=max_iterations)
                    if num_iterations == max_iterations:
                        raise ValueError("Solid to Solid surface tension solver failed to converge.")
                    st_coherent = (1/omega_coherent) * (R * T * log((1 - xB_i) / (1 - xB_alpha)) + L_ave*xB_i**2  - L_0_alpha*xB_alpha**2)
                    return st_coherent
                st_coherent = _calculate_coherent_solid_solid()
                return st_coherent + (z_V/0.47) * (st_incoherent - st_coherent)

            def _calculate_surface_tension_liquid_to_liquid(xB_alpha, xB_beta, phase_alpha="Liquid", phase_beta="Liquid"):
                f = 1.0
                omega_A = _calculate_omega_for_surface_tension(T_dependent_parameters.v_mp[0, phases.index(phase_alpha)], f=1.0)
                omega_B = _calculate_omega_for_surface_tension(T_dependent_parameters.v_mp[1, phases.index(phase_beta)], f=1.0)
                R = 8.31446261815324
                xB = xB_alpha
                L_0 = T_dependent_parameters.L_ip[phase_alpha][0]
                def _update_xB_i(xB_i_prev):
                    dS = (R*T/omega_A)*log((1-xB_i_prev)/(1-xB)) - (R*T/omega_B)*log(xB_i_prev/xB)
                    dEx = L_0*((xB_i_prev**2 - xB**2)/omega_A +  (1-xB)**2/omega_B)
                    return 1 - sqrt(dS + dEx)
                max_iterations = 100
                xB_i, num_iterations = self._generic_iterative_loop_one_variable(
                    initial_guess=xB,
                    update_function=_update_xB_i,
                    tol=1e-6,
                    max_iterations=max_iterations)
                if num_iterations == max_iterations:
                    raise ValueError("Liquid to Liquid surface tension solver failed to converge.")
                st_chem_calc = lambda L: (R*T/omega_A)*log((1-xB_i)/(1-xB)) + (L/omega_A) * (xB_i**2 - xB**2)
                st_chem = st_chem_calc(L_0)
                st_en = 2.9*T*(1/omega_A + 1/omega_B)
                st_chem_0 = st_chem_calc(self.interaction_data.phases["Liquid"].Li[0](0)) # type: ignore
                return st_chem*(1 + st_en/st_chem_0)

            def _calculate_surface_tension_solid_to_liquid(xB_solid, xB_liquid, phase_solid, phase_liquid="Liquid"):
                k = 0.9738
                f = 1.045
                v_A = T_dependent_parameters.v_mp[0, phases.index(phase_solid)]*0.5 + T_dependent_parameters.v_mp[0, phases.index(phase_liquid)]*0.5
                v_B = T_dependent_parameters.v_mp[1, phases.index(phase_solid)]*0.5 + T_dependent_parameters.v_mp[1, phases.index(phase_liquid)]*0.5
                omega_A = _calculate_omega_for_surface_tension(v_A, f)
                omega_B = _calculate_omega_for_surface_tension(v_B, f)
                dG_A_b = _calculate_G_excess_for_surface_tension_same_phase(xB_liquid, k, phase_liquid, is_dG_A=True)
                dG_B_b = _calculate_G_excess_for_surface_tension_same_phase(xB_liquid, k, phase_liquid, is_dG_A=False)
                R = 8.31446261815324
                dst_0 = 0.15*(T_dependent_parameters.st_mp[0, phases.index(phase_solid)] - T_dependent_parameters.st_mp[1, phases.index(phase_solid)])
                
                
                def update_xB_i(xB_i_prev):
                    dG_A_i = _calculate_G_excess_for_surface_tension_different_phases_interface(xB_i_prev, k, phase_solid, phase_liquid, is_dG_A=True)
                    dG_B_i = _calculate_G_excess_for_surface_tension_different_phases_interface(xB_i_prev, k, phase_solid, phase_liquid, is_dG_A=False)
                    return xB_liquid * exp((dst_0 + R * T * log((1 - xB_i_prev) / (1 - xB_liquid)) / omega_A + (dG_A_i - dG_A_b) / omega_A - (dG_B_i - dG_B_b) / omega_B) * omega_B / (R * T))
                max_iterations = 100
                xB_i, num_iterations = self._generic_iterative_loop_one_variable(
                    initial_guess=0.5*(xB_solid + xB_liquid),
                    update_function=update_xB_i,
                    tol=1e-6,
                    max_iterations=max_iterations)
                if num_iterations == max_iterations:
                    raise ValueError("Solid to Liquid surface tension solver failed to converge.")
                dG_A_i = _calculate_G_excess_for_surface_tension_different_phases_interface(xB_i, k, phase_solid, phase_liquid, is_dG_A=True)
                return T_dependent_parameters.st_mp[0, phases.index(phase_solid)] + ( R * T / omega_A) * log((1 - xB_i) / (1 - xB_liquid)) + (dG_A_i - dG_A_b) / omega_A
                
            
            
            match (phase_alpha, phase_beta):
                case ("Liquid", "Liquid"):
                    st = _calculate_surface_tension_liquid_to_liquid(xB_alpha, xB_beta)
                case ("FCC", None) | ("Liquid", None):
                    st = _calculate_surface_tension_alloy_to_vacuum(xB_alpha, phase_alpha)
                case (p_a, p_b) if p_b is not None and "Liquid" in (p_a, p_b) and p_a != p_b:
                    st = _calculate_surface_tension_solid_to_liquid(xB_alpha if p_a != "Liquid" else xB_beta, xB_beta if p_b != "Liquid" else xB_alpha, phase_alpha if p_a != "Liquid" else phase_beta, phase_beta if p_b == "Liquid" else phase_alpha)
                case (p_a, p_b) if p_b is not None and "Liquid" not in (p_a, p_b):
                    st = _calculate_surface_tension_solid_to_solid(xB_alpha, xB_beta, phase_alpha, phase_beta)
            return st
            
        spheric_r_vals = _calculate_spheric_Janus_geo()
        # st_a = _calculate_surface_tension_val(x_mp[1,0], skin.xB if skin.exists else None, phases[0], phases[2] if skin.exists else None, T, vals)
        # st_b = _calculate_surface_tension_val(x_mp[1,1], skin.xB if skin.exists else None, phases[1], phases[2] if skin.exists else None, T, vals)
        # st_i = _calculate_surface_tension_val(x_mp[1,0], x_mp[1,1], phases[0], phases[1])


        # Calculate spheric geo (initial guess + useful later if fails)
        # Calculate the surface tensions based on Kaptay and db vals - NOTE THAT IF SKIN EXISTS, VALUES ARE FOR OUTERPHASE
        # Solve h_alpha h_beta based (r_i is known based on them)
        # If fails, return spheric solution
        
        return np.array([1.0, 1.0])

    def _calc_core_shell_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T_dependent_parameters: TemperatureDependentVars,
    ) -> np.ndarray:
        """
        Calculates the geometry parameters for Core-Shell particles.
        """
        v_mp = T_dependent_parameters.v_mp
        V_alpha = np.sum(n_mp[:,0] * v_mp[:,0])
        V_core_and_shell = np.sum(n_mp[:,:2] * v_mp[:,:2])
        r_core = self.calc_r_from_V(V_alpha)
        r_core_and_shell = self.calc_r_from_V(V_core_and_shell)
        r_vals = np.array([r_core, r_core_and_shell]) ### r_vals def - Core-Shell no Skin
        return r_vals