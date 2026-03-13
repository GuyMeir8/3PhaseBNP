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

    def __init__(self, config: ThreePhaseConfiguration):
        self.config = config
        self.material_data = {}
        self.eps = 1e-9

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

        mat_names = self.config.materials
        g_mp_rows = []
        for mat_name in mat_names:
            row = []
            material = self.material_data[mat_name]
            for p in phases:
                phase_data = material.phases.get(p)
                row.append(phase_data.g0(T))
            g_mp_rows.append(row)

        return np.array(g_mp_rows)

    @staticmethod
    def calc_r_from_V(V): 
        if V < 0 or np.isnan(V):
            # Physical volume cannot be negative. This indicates an upstream logic error.
            raise RuntimeError(f"calc_r_from_V received invalid Volume: {V}")
        if V == 0:
            return 0.0
        return ((3 * V) / (4 * np.pi)) ** (1 / 3)

    def _get_v_mp(
            self,
            T: float,
            phases: Tuple[str, ...],
    ) -> np.ndarray:

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

    def _update_phases_based_on_skin(self, primary_phases, skin_val):
        skin = skin_class(skin_val)
        if skin.exists:
            phases = primary_phases + ("Liquid",)
        else:
            phases = primary_phases
        return phases, skin

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
        
        if xB_total == 0.5: xB_total = 0.5 - self.eps
        try:
            phases, skin = self._update_phases_based_on_skin(primary_phases, skin_val)
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
            G_ideal = self._calc_G_ideal(n_mp, x_mp, T, phases, T_dependent_parameters)
            G_excess = self._calc_G_excess(n_mp, x_mp, T, phases, T_dependent_parameters)
            G_surface = self._calc_G_surface(x_mp, r_vals, T, phases, geometry_type, skin, T_dependent_parameters)

            return G_ideal + G_excess + G_surface # type: ignore
            
        except (ValueError, OverflowError, ZeroDivisionError): # Anytime something returns "no solution" or diverges
            return 1.0
    
    def _calc_G_surface(self, x_mp, r_vals, T, phases, geometry_type, skin, T_dependent_parameters):
        A_Janus_out = lambda r, cos_theta : 2*np.pi*(r**2)*(1-cos_theta)
        match (geometry_type, skin.exists):
            case ("Janus", False):              
                st_alpha_vac = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,0],
                    xB_beta=None,
                    phase_alpha=phases[0],
                    phase_beta=None,
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                st_beta_vac = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,1],
                    xB_beta=None,
                    phase_alpha=phases[1],
                    phase_beta=None,
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                st_alpha_beta = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,0],
                    xB_beta=x_mp[1,1],
                    phase_alpha=phases[0],
                    phase_beta=phases[1],
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                r_alpha = r_vals[0]
                r_beta = r_vals[1]
                r_interface = r_alpha*sqrt(1-r_vals[2]**2)
                A_alpha_vac =A_Janus_out(r_alpha, r_vals[2])
                cos_theta_beta = sqrt(1-(r_interface/r_beta)**2)
                A_beta_out = A_Janus_out(r_beta, cos_theta_beta)
                A_alpha_beta = np.pi*r_interface**2
                return st_alpha_vac*A_alpha_vac + st_beta_vac*A_beta_out + st_alpha_beta*A_alpha_beta
            
            case ("Janus", True):
                skin_thickness = self._calc_skin_thickness(skin)
                r_alpha_no_skin = r_vals[0] - skin_thickness
                r_beta_no_skin = r_vals[1] - skin_thickness
                r_interface_alpha_beta = r_alpha_no_skin*sqrt(1-r_vals[2]**2)
                st_alpha_skin = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,0],
                    xB_beta=x_mp[1,2],
                    phase_alpha=phases[0],
                    phase_beta=phases[2],
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                st_beta_skin = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,1],
                    xB_beta=x_mp[1,2],
                    phase_alpha=phases[1],
                    phase_beta=phases[2],
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                st_alpha_beta = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,0],
                    xB_beta=x_mp[1,1],
                    phase_alpha=phases[0],
                    phase_beta=phases[1],
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                A_alpha_skin = A_Janus_out(r_alpha_no_skin, r_vals[2])
                cos_theta_beta = sqrt(1-(r_interface_alpha_beta/r_beta_no_skin)**2)
                A_beta_skin = A_Janus_out(r_beta_no_skin, cos_theta_beta)
                A_alpha_beta = np.pi*r_interface_alpha_beta**2
                G_surf_alpha_skin = st_alpha_skin*A_alpha_skin
                G_surf_beta_skin = st_beta_skin*A_beta_skin
                G_surf_alpha_beta = st_alpha_beta*A_alpha_beta

                A_skin = A_Janus_out(r_vals[0], r_vals[2]) + A_Janus_out(r_vals[1], cos_theta_beta)
                st_skin_vac = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,2],
                    xB_beta=None,
                    phase_alpha=phases[2],
                    phase_beta=None,
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                G_surf_skin_vac = A_skin*st_skin_vac
                return G_surf_alpha_skin + G_surf_beta_skin + G_surf_alpha_beta + G_surf_skin_vac

            case ("Core Shell", False):
                A_alpha_beta = 4*np.pi*r_vals[0]**2
                st_alpha_beta = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,0],
                    xB_beta=x_mp[1,1],
                    phase_alpha=phases[0],
                    phase_beta=phases[1],
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                A_beta_vac = 4*np.pi*r_vals[1]**2
                st_beta_vac = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,1],
                    xB_beta=None,
                    phase_alpha=phases[1],
                    phase_beta=None,
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                return st_alpha_beta*A_alpha_beta + st_beta_vac*A_beta_vac

            case ("Core Shell", True):
                A_alpha_beta = 4*np.pi*r_vals[0]**2
                st_alpha_beta = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,0],
                    xB_beta=x_mp[1,1],
                    phase_alpha=phases[0],
                    phase_beta=phases[1],
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                A_beta_skin = 4*np.pi*r_vals[1]**2
                st_beta_skin = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,1],
                    xB_beta=x_mp[1,2],
                    phase_alpha=phases[1],
                    phase_beta=phases[2],
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )
                A_skin_vac = 4*np.pi*r_vals[2]**2
                st_skin_vac = self._calculate_surface_tension(
                    xB_alpha=x_mp[1,2],
                    xB_beta=None,
                    phase_alpha=phases[2],
                    phase_beta=None,
                    T=T,
                    phases=phases,
                    T_dependent_parameters=T_dependent_parameters,
                )

                return st_alpha_beta*A_alpha_beta + st_beta_skin*A_beta_skin + st_skin_vac*A_skin_vac

    @staticmethod
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
    
    def _calc_skin_thickness(self, skin_data):
        x_skin_m = np.array([[1-skin_data.xB], [skin_data.xB]])
        mats = self.config.materials
        pure_mat_thickness_m = np.array([[self.material_data[m].atomic_radius] for m in mats])
        return np.sum(x_skin_m * pure_mat_thickness_m * 2)

    @staticmethod
    def calc_V_from_r(r): return (4 * np.pi * r**3) / 3
    
    def _calc_initial_n_skin(self, v_skin, n_total, skin_thickness):
        V_initial = n_total*v_skin
        r_initial = self.calc_r_from_V(V_initial)
        r_wo_skin = r_initial - skin_thickness
        V_wo_skin = self.calc_V_from_r(r_wo_skin)
        V_skin_initial = V_initial - V_wo_skin        
        n_skin = V_skin_initial / v_skin
        return n_skin

    @staticmethod
    def _calc_V_for_given_Janus_geo(r_alpha, r_beta, cos_theta_alpha):
        V_calc = lambda r, cos_theta : np.pi*(r**3)*(2+cos_theta)*((1-cos_theta)**2)/3
        V_alpha = V_calc(r_alpha, cos_theta_alpha)
        V_beta = V_calc(r_beta, 1-cos_theta_alpha)
        return V_alpha+V_beta

    def _calc_mole_skin_next_guess(
            self, 
            n_skin, 
            n_A_total, 
            n_B_total,
            A_ratio_alpha, 
            B_ratio_alpha,
            phases, 
            geometry_type, 
            skin_data, 
            T_dependent_parameters,
            T,
            skin_thickness,
            v_skin
            ):
        
        n_A_no_skin = n_A_total - n_skin*(1-skin_data.xB)
        n_B_no_skin = n_B_total - n_skin*skin_data.xB

        if n_A_no_skin < 0 or n_B_no_skin < 0:
            return 1e9 # Return a large value to discourage the solver from this path

        n_A_alpha = A_ratio_alpha * n_A_no_skin
        n_B_alpha = B_ratio_alpha * n_B_no_skin
        n_mp, x_mp = self._calc_generic_split(n_A_alpha, n_B_alpha, n_A_no_skin, n_B_no_skin)
        match geometry_type:
            case "Janus":
                r_vals = self._calc_Janus_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T,
                        T_dependent_parameters,
                        skin_data
                    )
                V_no_skin = self._calc_V_for_given_Janus_geo(r_vals[0], r_vals[1], r_vals[2])
                V_w_skin = self._calc_V_for_given_Janus_geo(
                    r_vals[0] + skin_thickness,
                    r_vals[1] + skin_thickness,
                    r_vals[2]
                )
                V_skin = V_w_skin - V_no_skin
                n_skin = V_skin / v_skin
                return n_skin
                
            case "Core Shell":
                r_vals = self._calc_core_shell_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T_dependent_parameters,
                    )
                V_no_skin = self.calc_V_from_r(r_vals[1])
                V_w_skin = self.calc_V_from_r(r_vals[1] + skin_thickness)
                V_skin = V_w_skin - V_no_skin
                n_skin = V_skin / v_skin
                return n_skin
    
    def _calc_nx_after_skin_is_found(self, n_skin, skin, n_A_total, n_B_total, A_ratio_alpha, B_ratio_alpha, phases, geometry_type, T, T_dependent_parameters, skin_thickness):
        n_A_skin = n_skin * (1 - skin.xB)
        n_B_skin = n_skin * skin.xB

        n_A_no_skin = n_A_total - n_A_skin
        n_B_no_skin = n_B_total - n_B_skin

        if n_A_no_skin < 0 or n_B_no_skin < 0:
            raise ValueError("Skin requires more material than available.")

        n_A_alpha = A_ratio_alpha * n_A_no_skin
        n_B_alpha = B_ratio_alpha * n_B_no_skin
        
        n_mp, x_mp = self._calc_generic_split(n_A_alpha, n_B_alpha, n_A_no_skin, n_B_no_skin)
        
        match geometry_type:
            case "Janus":
                r_vals = self._calc_Janus_geometry_for_known_nx(
                    n_mp, x_mp, phases, T, T_dependent_parameters, skin
                )
                r_vals = np.array([
                    r_vals[0] + skin_thickness,
                    r_vals[1] + skin_thickness,
                    r_vals[2]
                ]) # Janus r_vals = Only values with skins

            case "Core Shell":
                r_vals = self._calc_core_shell_geometry_for_known_nx(
                    n_mp, x_mp, phases, T_dependent_parameters
                )
                r_vals = np.array([
                    r_vals[0],
                    r_vals[1],
                    r_vals[1] + skin_thickness
                ]) # Core Shell r_vals - alpha, beta, everything
        

        n_skin_col = np.array([[n_A_skin], [n_B_skin]])
        x_skin_col = np.array([[1 - skin.xB], [skin.xB]])
        
        n_mp = np.hstack((n_mp, n_skin_col))
        x_mp = np.hstack((x_mp, x_skin_col))

        return n_mp, x_mp, r_vals

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

        n_A_total = n_total * (1 - xB_total)
        n_B_total = n_total * xB_total
        
        if skin.exists:
            skin_thickness = self._calc_skin_thickness(skin)
            x_skin_m = np.array([[1-skin.xB], [skin.xB]])
            v_skin = np.sum(x_skin_m * T_dependent_parameters.v_mp[:,2])
            n_skin = self._calc_initial_n_skin(v_skin, n_total, skin_thickness)
            eps_for_nano = 1e-4
            n_skin, success = self.hybrid_solver_one_variable(
                min_val = n_total*eps_for_nano,
                max_val = n_total*(1-eps_for_nano),
                initial_guess = n_skin,
                update_function = lambda ns: self._calc_mole_skin_next_guess(
                    ns, n_A_total, n_B_total, A_ratio_alpha, B_ratio_alpha,
                    phases, geometry_type, skin, T_dependent_parameters, T,
                    skin_thickness, v_skin
                ),  # type: ignore
                tol= eps_for_nano*n_total,
            )

            if not success:
                raise ValueError("Mole split solver failed to converge.")
            
            n_mp, x_mp, r_vals = self._calc_nx_after_skin_is_found(n_skin, skin, n_A_total, n_B_total, A_ratio_alpha, B_ratio_alpha, phases, geometry_type, T, T_dependent_parameters, skin_thickness)


        else:
            n_A_alpha = A_ratio_alpha * n_A_total
            n_B_alpha = B_ratio_alpha * n_B_total
            n_mp, x_mp = self._calc_generic_split(n_A_alpha, n_B_alpha, n_A_total, n_B_total)
            
            match geometry_type:
                case "Janus":
                    r_vals = self._calc_Janus_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T,
                        T_dependent_parameters,
                        skin
                    )
                case "Core Shell":
                    r_vals = self._calc_core_shell_geometry_for_known_nx(
                        n_mp,
                        x_mp,
                        phases,
                        T_dependent_parameters,
                    )
        x_mp = np.clip(x_mp, self.eps, 1.0 - self.eps)
        return n_mp, x_mp, r_vals

    @staticmethod
    def real_from_roots(roots_prereal):
            r = roots_prereal
            r = r[np.abs(r.imag) < 1e-9].real
            r = r[r > 0]
            return r

    def _calc_Janus_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T: float,
            T_dependent_parameters: TemperatureDependentVars,
            skin: skin_class
    ) -> np.ndarray:
        
        r_spheric = self._calculate_spheric_Janus_geo(n_mp, T_dependent_parameters)
        
        h_alpha_spheric = r_spheric[0] * (1 - r_spheric[2])
        h_beta_spheric = r_spheric[1] * (1 + r_spheric[2])

        st_alpha_beta = self._calculate_surface_tension(
            xB_alpha=x_mp[1,0],
            xB_beta=x_mp[1,1],
            phase_alpha=phases[0],
            phase_beta=phases[1],
            T=T,
            phases=phases,
            T_dependent_parameters=T_dependent_parameters,
        )

        outer_phase = phases[2] if skin.exists else None
        xB_outer = skin.xB if skin.exists else None

        st_alpha_out = self._calculate_surface_tension(
            xB_alpha=x_mp[1,0],
            xB_beta=xB_outer,
            phase_alpha=phases[0],
            phase_beta=outer_phase,
            T=T,
            phases=phases,
            T_dependent_parameters=T_dependent_parameters,
        )

        st_beta_out = self._calculate_surface_tension(
            xB_alpha=x_mp[1,1],
            xB_beta=xB_outer,
            phase_alpha=phases[1],
            phase_beta=outer_phase,
            T=T,
            phases=phases,
            T_dependent_parameters=T_dependent_parameters,
        )

        a = st_alpha_out + st_beta_out + st_alpha_beta
        b = lambda h2_alpha, h2_beta : (h2_alpha+h2_beta)*st_alpha_beta - (st_alpha_out - st_beta_out)*(h2_alpha-h2_beta)*st_alpha_beta
        c = lambda h2_alpha, h2_beta : h2_alpha*h2_beta*(st_alpha_beta - st_alpha_out - st_beta_out)

        V_alpha =np.sum( n_mp[:,0] * T_dependent_parameters.v_mp[:,0])
        V_beta = np.sum( n_mp[:,1] * T_dependent_parameters.v_mp[:,1])

        def _calc_r2_i_roots(h_alpha, h_beta):
            h2_alpha = h_alpha**2
            h2_beta = h_beta**2
            b_curr = b(h2_alpha, h2_beta)
            c_curr = c(h2_alpha, h2_beta)
            r2_i = np.roots([a, b_curr, c_curr])
            return r2_i

        def equations(vars):
            h_alpha, h_beta = vars
            r2_i =_calc_r2_i_roots(h_alpha, h_beta)
            r2_i = self.real_from_roots(r2_i)
            if len(r2_i) == 0:
                return [1/self.eps, 1/self.eps]
            
            # Pick the root that corresponds to a radius closest to the spheric approximation
            # This prevents jumping to unphysical branches of the solution
            target_r2 = (r_spheric[0] * sqrt(1 - r_spheric[2]**2))**2
            best_idx = np.argmin(np.abs(r2_i - target_r2))
            r2_val = r2_i[best_idx]
            
            eq_def = lambda h, V : np.pi*h*(3*r2_val + h**2) / 6 - V
            eq_alpha = eq_def(h_alpha, V_alpha)
            eq_beta = eq_def(h_beta, V_beta)
            return [eq_alpha, eq_beta]
        
        sol, _, ier, _ = optimize.fsolve(equations, [h_alpha_spheric, h_beta_spheric], full_output=True)
        
        if ier != 1: return r_spheric
        
        h_alpha, h_beta = sol
        
        a_2_final = _calc_r2_i_roots(h_alpha, h_beta)
        a_2_final = self.real_from_roots(a_2_final)

        if len(a_2_final) != 1: return r_spheric
        a_final = sqrt(a_2_final[0])

        # Correct formula for Sphere Radius R given cap height h and base radius a:
        # R = (h^2 + a^2) / (2h)
        # Guard against h=0 to prevent division by zero
        if h_alpha < self.eps or h_beta < self.eps: return r_spheric

        r_alpha = (h_alpha**2 + a_final**2) / (2 * h_alpha)
        r_beta = (h_beta**2 + a_final**2) / (2 * h_beta)

        # SANITY CHECK: If force-balance solver diverged to huge radii (flat interface) 
        # or returned NaNs, revert to the robust spheric approximation.
        if (np.isnan(r_alpha) or np.isnan(r_beta) or 
            r_alpha > 10 * r_spheric[0] or r_beta > 10 * r_spheric[1]):
            return r_spheric

        # Correct formula for cos(theta): h = R(1 - cos_theta) -> cos_theta = 1 - h/R
        cos_theta_alpha = 1.0 - h_alpha / r_alpha
        
        return np.array([r_alpha, r_beta, cos_theta_alpha])

    def _calculate_spheric_Janus_geo(self, n_mp, T_dependent_parameters):
        V_alpha = np.sum(n_mp[:,0] * T_dependent_parameters.v_mp[:,0])
        V_beta = np.sum(n_mp[:,1] * T_dependent_parameters.v_mp[:,1])

        # Guard against negative volumes (unphysical input)
        # Guard against negative volumes (unphysical input) which imply
        # negative moles or invalid molar volumes from the database.
        if V_alpha <= 0 or np.isnan(V_alpha):
            raise RuntimeError(f"Invalid V_alpha in spheric Janus calc: {V_alpha}")
        if V_beta <= 0 or np.isnan(V_beta):
            raise RuntimeError(f"Invalid V_beta in spheric Janus calc: {V_beta}")

        V_ratio = V_alpha / V_beta

        # The volume ratio Q = V_alpha / V_beta for a spherical particle cut by a plane satisfies:
        # Q = (2 - 3c + c^3) / (2 + 3c - c^3)
        # This rearranges to the cubic polynomial:
        # c^3 - 3c + 2 * (1 - Q) / (1 + Q) = 0
        # We solve this directly instead of using the unstable iterative solver.
        
        K = 2 * (1 - V_ratio) / (1 + V_ratio)
        roots = np.roots([1, 0, -3, K])
        
        # We need the real root in the range [-1, 1]
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        valid_roots = real_roots[(real_roots >= -1.0 - 1e-9) & (real_roots <= 1.0 + 1e-9)]
        
        # If multiple roots exist (unlikely in this range), take the one closest to 0
        if len(valid_roots) > 0:
            cos_theta_alpha = valid_roots[np.argmin(np.abs(valid_roots))]
        else:
            # Fallback for numerical edge cases
            cos_theta_alpha = 1.0 if K > 0 else -1.0

        # Clip to prevent division by zero in radius calculation
        cos_theta_alpha = np.clip(cos_theta_alpha, -1.0 + self.eps, 1.0 - self.eps)

        r_alpha = ((3 * V_alpha) / (np.pi * (2 + cos_theta_alpha) * (1 - cos_theta_alpha) ** 2)) ** (1/3)
        r_beta = ((3 * V_beta) / (np.pi * (2 - cos_theta_alpha) * (1 + cos_theta_alpha) ** 2)) ** (1/3)
        
        return np.array([r_alpha, r_beta, cos_theta_alpha])
        
    def _calc_G_ideal(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            T: float,
            phases: Tuple[str, ...],
            T_dependent_parameters: TemperatureDependentVars,
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
        g_mp = T_dependent_parameters.g_mp

        for phase_idx in range(num_phases):
            n_phase = np.sum(n_mp[:, phase_idx])
            g_ideal_phase = x_mp[:,phase_idx] * g_mp[:,phase_idx] + R * T * x_mp[:,phase_idx] * np.log(x_mp[:,phase_idx])
            G_ideal += n_phase * np.sum(g_ideal_phase)
        return G_ideal

    def _calc_G_excess(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            T: float,
            phases: Tuple[str, ...],
            T_dependent_parameters: TemperatureDependentVars,
    ) -> float:
        """
        Calculates the excess Gibbs free energy for the system.
        """
        G_excess = 0.0

        for i, phase in enumerate(phases):
            n_phase = np.sum(n_mp[:, i])
            xA = x_mp[0, i]
            xB = x_mp[1, i]
            L_values = T_dependent_parameters.L_ip[phase]
            interaction_sum = np.sum([L * ((xA - xB) ** k) for k, L in enumerate(L_values)])
            G_excess += n_phase * xA * xB * interaction_sum
        return G_excess
    
    def generic_iterative_loop_one_variable(
            self,
            initial_guess: float,
            update_function: Callable[[float], float],
            tol: float = 1e-6,
            max_iterations: int = 100,
    ) -> Tuple[float, bool]:

        was_loop_successful = False
        var_current = initial_guess
        for _ in range(max_iterations):
            var_next = update_function(var_current)
            if abs(var_next - var_current) < tol:
                was_loop_successful = True
                break
            var_current = var_next*0.9 + var_current*0.1 

        return (var_current, was_loop_successful)
    
    def generic_bisection_loop_one_variable(
            self,
            min_val: float,
            max_val: float,
            initial_guess: float,
            update_function: Callable[[float], float],
            tol: float = 1e-6,
            max_iterations: int = 100,
    ) -> Tuple[float, bool]:
        
        was_successful = False

        def f(x):
            return x - update_function(x)

        y_min = f(min_val)
        if abs(y_min) < tol: return (min_val, True)

        y_max = f(max_val)
        if abs(y_max) < tol: return (max_val, True)

        if y_min * y_max > 0:
            return (initial_guess, was_successful)

        low = min_val
        high = max_val
        
        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            y_mid = f(mid)
            
            if abs(y_mid) < tol or abs(high - low) < tol:
                was_successful = True
                return (mid, was_successful)
            
            if y_min * y_mid < 0:
                high = mid
                y_max = y_mid
                
            else:
                low = mid
                y_min = y_mid
                
        return ((low + high) / 2.0, was_successful)

    def hybrid_solver_one_variable(
            self,
            min_val: float,
            max_val: float,
            initial_guess: float,
            update_function: Callable[[float], float],
            tol: float = 1e-6,
            max_iterations: int = 100,
    ) -> Tuple[float, bool]:
        
        val, success = self.generic_iterative_loop_one_variable(
            initial_guess=initial_guess,
            update_function=update_function,
            tol=tol,
            max_iterations=max_iterations
        )
        
        if success and val > self.eps + min_val and val < max_val - self.eps:
            return val, success
            
        return self.generic_bisection_loop_one_variable(
            min_val=min_val,
            max_val=max_val,
            initial_guess=initial_guess,
            update_function=update_function,
            tol=tol,
            max_iterations=max_iterations
        )

    @staticmethod
    def _calculate_G_excess_for_surface_tension_same_phase(xB, k, phase, is_dG_A, T_dependent_parameters):
        if not is_dG_A:
            xB = 1 - xB
        L_params = T_dependent_parameters.L_ip[phase]
        return k*(xB**2)*(L_params[0]+L_params[1]*(3-4*xB))

    @staticmethod
    def _calculate_omega_for_surface_tension(v_i, f):
        # Molar volume must be positive. A negative value here means the 
        # database returned a negative volume for the given Temperature, 
        # or the phase mixture logic is incorrect.
        if v_i <= 0 or np.isnan(v_i):
            raise RuntimeError(f"Invalid molar volume v_i: {v_i}")
        return f*(v_i**(2/3))*(6.02214e23)**(1/3)

    @staticmethod
    def _calculate_G_excess_for_surface_tension_different_phases_interface(xB_i, k, phase_alpha, phase_beta, is_dG_A, T_dependent_parameters):
        if not is_dG_A:
            xB_i = 1 - xB_i
        L_alpha_0 = T_dependent_parameters.L_ip[phase_alpha][0]
        L_beta_0 = T_dependent_parameters.L_ip[phase_beta][0]
        return k*(xB_i**2)*(0.5*(L_alpha_0 + L_beta_0))

    def _calculate_surface_tension_alloy_to_vacuum(self, xB, phase, T, phases, T_dependent_parameters):
        k = 0.804 if phase == "Liquid" else 0.75
        f = 1.0 if phase == "Liquid" else 1.09 
        phase_num = phases.index(phase)
        v_A = T_dependent_parameters.v_mp[0, phase_num]
        v_B = T_dependent_parameters.v_mp[1, phase_num]
        omega_A = self._calculate_omega_for_surface_tension(v_A, f)
        omega_B = self._calculate_omega_for_surface_tension(v_B, f)
        dG_A_b = self._calculate_G_excess_for_surface_tension_same_phase(xB, k, phase, True, T_dependent_parameters)
        dG_B_b = self._calculate_G_excess_for_surface_tension_same_phase(xB, k, phase, False, T_dependent_parameters)
        R = 8.31446261815324 
        dS_b = R*T*( log(xB)/omega_B - log(1-xB)/omega_A)
        dG_b = dG_B_b/omega_B - dG_A_b/omega_A
        dst_0 = T_dependent_parameters.st_mp[0, phase_num] - T_dependent_parameters.st_mp[1, phase_num]
        d_const = dst_0 + dS_b + dG_b

        def update_xB_i(xB_i_prev):
            # Guard against solver overshoot causing domain errors in log
            xB_i_prev = np.clip(xB_i_prev, self.eps, 1.0 - self.eps)
            dG_A_i = self._calculate_G_excess_for_surface_tension_same_phase(xB_i_prev, k, phase, True, T_dependent_parameters)
            dG_B_i = self._calculate_G_excess_for_surface_tension_same_phase(xB_i_prev, k, phase, False, T_dependent_parameters)
            x_A_i_prev = 1 - xB_i_prev
            dS_i_prev = R*T*log(x_A_i_prev)/omega_A
            dG_i = dG_A_i/omega_A - dG_B_i/omega_B
            return exp( (omega_B / (R*T)) * (d_const + dS_i_prev + dG_i) )

        xB_i, success = self.hybrid_solver_one_variable(
            min_val=self.eps,
            max_val=1.0-self.eps,
            initial_guess=xB,
            update_function=update_xB_i)

        if not success:
            raise ValueError("Alloy to Vacuum surface tension solver failed to converge.")
        
        dG_A_i = self._calculate_G_excess_for_surface_tension_same_phase(xB_i, k, phase, True, T_dependent_parameters)
        st_vac_alloy = T_dependent_parameters.st_mp[0, phase_num] + ( R * T / omega_A) * log((1 - xB_i) / (1 - xB)) + (dG_A_i - dG_A_b) / omega_A
        return st_vac_alloy

    def _calculate_surface_tension_solid_to_solid(self, xB_alpha, xB_beta, phase_alpha, phase_beta, T, phases, T_dependent_parameters):
        x_vec_alpha = np.array([1.0 - xB_alpha, xB_alpha])
        x_vec_beta = np.array([1.0 - xB_beta, xB_beta])
        
        v_alpha = np.sum(T_dependent_parameters.v_mp[:, phases.index(phase_alpha)] * x_vec_alpha)
        v_beta = np.sum(T_dependent_parameters.v_mp[:, phases.index(phase_beta)] * x_vec_beta)
        z_V = v_alpha / v_beta - 1 if v_alpha > v_beta else v_beta / v_alpha - 1
        R = 8.31446261815324
        st_A_solid_solid = T_dependent_parameters.st_mp[0, phases.index(phase_alpha)] * (1/3)
        st_B_solid_solid = T_dependent_parameters.st_mp[1, phases.index(phase_beta)] * (1/3)
        
        def _calculate_incoherent_solid_solid():
            k_incoherent = 0.917
            f_incoherent = 1.045
            omega_A_incoherent = self._calculate_omega_for_surface_tension(v_alpha, f_incoherent)
            omega_B_incoherent = self._calculate_omega_for_surface_tension(v_beta, f_incoherent)
            L_0_FCC = T_dependent_parameters.L_ip["FCC"][0]
            A_const = st_A_solid_solid - (R*T/omega_A_incoherent) * log(1 - xB_alpha) - L_0_FCC * xB_alpha**2 / omega_A_incoherent
            B_const = st_B_solid_solid - (R*T/omega_B_incoherent) * log(xB_alpha) - L_0_FCC * (1 - xB_alpha)**2 / omega_B_incoherent
            d_const = A_const - B_const
            def update_xB_i(xB_i_prev):
                xB_i_prev = np.clip(xB_i_prev, self.eps, 1.0 - self.eps)
                dS = (R*T/omega_A_incoherent)*log(1-xB_i_prev)
                dG_Ex = L_0_FCC*k_incoherent*(xB_i_prev**2/omega_A_incoherent - (1-xB_i_prev)**2/omega_B_incoherent)
                return exp((omega_B_incoherent/(R*T))*(d_const+dS+dG_Ex))
            xB_i, success = self.hybrid_solver_one_variable(
                min_val=self.eps,
                max_val=1.0-self.eps,
                initial_guess=0.5*(xB_alpha + xB_beta),
                update_function=update_xB_i
                )
            if not success:
                raise ValueError("Solid to Solid surface tension solver failed to converge.")
            return st_A_solid_solid + (R*T/omega_A_incoherent)*log((1-xB_i)/(1-xB_alpha))+(L_0_FCC/omega_A_incoherent)*((k_incoherent*xB_i**2 - xB_alpha**2))
        st_incoherent = _calculate_incoherent_solid_solid()
        if z_V >= 0.47:
            return st_incoherent
        def _calculate_coherent_solid_solid():
            f = 1.09
            omega_coherent = self._calculate_omega_for_surface_tension(T_dependent_parameters.v_mp[0, phases.index(phase_alpha)], f)
            L_0_alpha = T_dependent_parameters.L_ip[phase_alpha][0]
            L_0_beta = T_dependent_parameters.L_ip[phase_beta][0]
            L_ave = 0.5*(L_0_alpha + L_0_beta)
            dS_const = R*T*log(xB_alpha/(1-xB_alpha))
            dEx_const = L_0_alpha*(1 - 2*xB_alpha)
            d_const = dS_const + dEx_const
            def update_xB_i(xB_i_prev):
                xB_i_prev = np.clip(xB_i_prev, self.eps, 1.0 - self.eps)
                return exp(d_const/(R*T) + log(1-xB_i_prev) + (L_ave/(R*T))*(2*xB_i_prev - 1))
            xB_i, success = self.hybrid_solver_one_variable(
                min_val=self.eps,
                max_val=1.0-self.eps,
                initial_guess=0.5*(xB_alpha + xB_beta),
                update_function=update_xB_i,
                )
            if not success:
                raise ValueError("Solid to Solid surface tension solver failed to converge.")
            st_coherent = (1/omega_coherent) * (R * T * log((1 - xB_i) / (1 - xB_alpha)) + L_ave*xB_i**2  - L_0_alpha*xB_alpha**2)
            return st_coherent
        st_coherent = _calculate_coherent_solid_solid()
        return st_coherent + (z_V/0.47) * (st_incoherent - st_coherent)

    def _calculate_surface_tension_liquid_to_liquid(self, xB_alpha, xB_beta, T, phases, T_dependent_parameters, phase_alpha="Liquid", phase_beta="Liquid"):
        f = 1.0
        omega_A = self._calculate_omega_for_surface_tension(T_dependent_parameters.v_mp[0, phases.index(phase_alpha)], f=1.0)
        omega_B = self._calculate_omega_for_surface_tension(T_dependent_parameters.v_mp[1, phases.index(phase_beta)], f=1.0)
        R = 8.31446261815324
        xB = xB_alpha
        L_0 = T_dependent_parameters.L_ip[phase_alpha][0]
        min_st_val = 1e-2
        if L_0<0 or T>(L_0/(2*R)) or set(self.config.materials) == {"Ag", "Cu"}: return min_st_val 
        
        dS_const = R*T*(log(xB_alpha)/omega_B - log(1-xB_alpha)/omega_A)
        dEx_const = L_0*(((1-xB_alpha)**2)/omega_B + (xB_alpha**2)/omega_A)
        d_const = dS_const + dEx_const
        def _update_xB_i(xB_i_prev):
            xB_i_prev = np.clip(xB_i_prev, self.eps, 1.0 - self.eps)
            dS_i = R*T*log(1-xB_i_prev)/omega_A
            dEx_i = L_0*(xB_i_prev**2/omega_A - (1-xB_i_prev)**2/omega_B)
            return exp((omega_B/(R*T))*(dS_i + dEx_i + d_const))
        xB_i, success = self.hybrid_solver_one_variable(
            min_val=self.eps,
            max_val=1.0-self.eps,
            initial_guess=(xB_alpha + xB_beta)/2,
            update_function=_update_xB_i
            )
        if not success:
            raise ValueError("Liquid to Liquid surface tension solver failed to converge.")
        st_chem_calc = lambda L, T_calc: (R*T_calc/omega_A)*log((1-xB_i)/(1-xB)) + (L/omega_A) * (xB_i**2 - xB**2)
        st_chem = st_chem_calc(L_0, T)
        st_en = 2.9*T*(1/omega_A + 1/omega_B)
        st_chem_0 = st_chem_calc(self.interaction_data.phases["Liquid"].Li[0](0), 0) # type: ignore
        st_final = st_chem*(1 + st_en/st_chem_0) 
        if st_final < min_st_val : return min_st_val 
        return st_final

    def _calculate_surface_tension_solid_to_liquid(self, xB_solid, xB_liquid, phase_solid, T, phases, T_dependent_parameters, phase_liquid="Liquid"):
        k = 0.9738
        f = 1.045
        v_A = T_dependent_parameters.v_mp[0, phases.index(phase_solid)]*0.5 + T_dependent_parameters.v_mp[0, phases.index(phase_liquid)]*0.5
        v_B = T_dependent_parameters.v_mp[1, phases.index(phase_solid)]*0.5 + T_dependent_parameters.v_mp[1, phases.index(phase_liquid)]*0.5
        omega_A = self._calculate_omega_for_surface_tension(v_A, f)
        omega_B = self._calculate_omega_for_surface_tension(v_B, f)
        dG_A_b = self._calculate_G_excess_for_surface_tension_same_phase(xB_liquid, k, phase_liquid, True, T_dependent_parameters)
        dG_B_b = self._calculate_G_excess_for_surface_tension_same_phase(xB_liquid, k, phase_liquid, False, T_dependent_parameters)
        R = 8.31446261815324
        dst_0 = 0.15*(T_dependent_parameters.st_mp[0, phases.index(phase_solid)] - T_dependent_parameters.st_mp[1, phases.index(phase_solid)])
        dG_Ex_b = dG_B_b/omega_B - dG_A_b/omega_A
        dS_b = R*T*(log(xB_liquid)/omega_B - log(1-xB_liquid)/omega_A)
        d_const = dst_0 + dS_b + dG_Ex_b
        
        def update_xB_i(xB_i_prev):
            xB_i_prev = np.clip(xB_i_prev, self.eps, 1.0 - self.eps)
            dG_A_i = self._calculate_G_excess_for_surface_tension_different_phases_interface(xB_i_prev, k, phase_solid, phase_liquid, True, T_dependent_parameters)
            dG_B_i = self._calculate_G_excess_for_surface_tension_different_phases_interface(xB_i_prev, k, phase_solid, phase_liquid, False, T_dependent_parameters)
            dG_i = dG_A_i/omega_A - dG_B_i/omega_B
            dS_i_prev = R*T*log(1-xB_i_prev)/omega_A
            return exp((d_const + dS_i_prev + dG_i) * (omega_B / (R * T)))
        xB_i, success = self.hybrid_solver_one_variable(
            min_val=self.eps,
            max_val=1.0-self.eps,
            initial_guess=0.5*(xB_solid + xB_liquid),
            update_function=update_xB_i,
            )
        if not success:
            raise ValueError("Solid to Liquid surface tension solver failed to converge.")
        dG_A_i = self._calculate_G_excess_for_surface_tension_different_phases_interface(xB_i, k, phase_solid, phase_liquid, True, T_dependent_parameters)
        return T_dependent_parameters.st_mp[0, phases.index(phase_solid)] + ( R * T / omega_A) * log((1 - xB_i) / (1 - xB_liquid)) + (dG_A_i - dG_A_b) / omega_A

    def _calculate_surface_tension(
            self,
            xB_alpha: float,
            xB_beta: Optional[float],
            phase_alpha: str,
            phase_beta: Optional[str],
            T: float,
            phases: Tuple[str, ...],
            T_dependent_parameters: TemperatureDependentVars,
    ) -> float:
        match (phase_alpha, phase_beta):
            case ("Liquid", "Liquid"):
                st = self._calculate_surface_tension_liquid_to_liquid(xB_alpha, xB_beta, T, phases, T_dependent_parameters)
            case ("FCC", None) | ("Liquid", None):
                st = self._calculate_surface_tension_alloy_to_vacuum(xB_alpha, phase_alpha, T, phases, T_dependent_parameters)
            case (p_a, p_b) if p_b is not None and "Liquid" in (p_a, p_b) and p_a != p_b:
                st = self._calculate_surface_tension_solid_to_liquid(xB_alpha if p_a != "Liquid" else xB_beta, xB_beta if p_b != "Liquid" else xB_alpha, phase_alpha if p_a != "Liquid" else phase_beta, T, phases, T_dependent_parameters, phase_liquid=phase_beta if p_b == "Liquid" else phase_alpha)
            case (p_a, p_b) if p_b is not None and "Liquid" not in (p_a, p_b):
                st = self._calculate_surface_tension_solid_to_solid(xB_alpha, xB_beta, phase_alpha, phase_beta, T, phases, T_dependent_parameters)
        return st

    def _calc_core_shell_geometry_for_known_nx(
            self,
            n_mp: np.ndarray,
            x_mp: np.ndarray,
            phases: Tuple[str, ...],
            T_dependent_parameters: TemperatureDependentVars,
    ) -> np.ndarray:

        v_mp = T_dependent_parameters.v_mp
        V_alpha = np.sum(n_mp[:,0] * v_mp[:,0])
        V_core_and_shell = np.sum(n_mp[:,:2] * v_mp[:,:2])
        r_core = self.calc_r_from_V(V_alpha)
        r_core_and_shell = self.calc_r_from_V(V_core_and_shell)
        r_vals = np.array([r_core, r_core_and_shell]) ### r_vals def - Core-Shell no Skin
        return r_vals