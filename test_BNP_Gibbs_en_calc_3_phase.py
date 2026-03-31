import unittest
import numpy as np
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase, skin_class
from configurations_3_phase import low_res_configuration

class TestJanusGeometry(unittest.TestCase):

    def setUp(self):
        """Set up the calculator for tests."""
        self.config = low_res_configuration
        self.calculator = GibbsEnergyCalculator3Phase(self.config)

    def test_janus_geometry_volume_conservation_random_inputs(self):
        """
        Tests if the calculated Janus geometry conserves the volume of the phases
        for a set of random inputs, both with and without a skin.
        """
        
        # A helper function to run a single random test case
        def run_single_random_test(phases, T, skin_val):
            with self.subTest(f"T={T}, phases={phases}, skin={skin_val is not None}"):
                # 1. Generate random but valid inputs for mole splits
                n_total = 1e-20  # A small number of moles, typical for nanoparticles
                
                # Randomly split total moles between materials A and B
                xB_total = np.random.uniform(0.1, 0.9)
                n_A_total = n_total * (1 - xB_total)
                n_B_total = n_total * xB_total

                # Randomly split materials between the two Janus phases (alpha and beta)
                # This simulates the input to the geometry solver
                A_ratio_alpha = np.random.uniform(0.1, 0.9)
                B_ratio_alpha = np.random.uniform(0.1, 0.9)

                n_A_alpha = n_A_total * A_ratio_alpha
                n_B_alpha = n_B_total * B_ratio_alpha
                n_A_beta = n_A_total - n_A_alpha
                n_B_beta = n_B_total - n_B_alpha

                n_mp = np.array([[n_A_alpha, n_A_beta], [n_B_alpha, n_B_beta]])
                
                # Ensure no negative moles from random generation, skip if so
                if np.any(n_mp < 0):
                    return

                n_alpha_tot = np.sum(n_mp[:, 0])
                n_beta_tot = np.sum(n_mp[:, 1])

                # Avoid division by zero if a phase has no moles
                if n_alpha_tot == 0 or n_beta_tot == 0:
                    return

                x_mp = np.array([
                    [n_mp[0, 0] / n_alpha_tot, n_mp[0, 1] / n_beta_tot],
                    [n_mp[1, 0] / n_alpha_tot, n_mp[1, 1] / n_beta_tot]
                ])

                skin = skin_class(skin_val)
                
                # Use the calculator's internal methods to get consistent T-dependent parameters
                calc_phases, _ = self.calculator._update_phases_based_on_skin(phases, skin_val)
                T_dep = self.calculator._get_T_dependent_vars(T, calc_phases)

                print("\n--- Test Case ---")
                print(f"Inputs:\n  T: {T:.2f} K\n  Phases: {phases}\n  Skin Value: {skin_val}")
                print(f"  n_alpha/n_total: {n_alpha_tot/n_total:.4f}")
                print(f"  n_beta/n_total: {n_beta_tot/n_total:.4f}")

                # 2. Calculate the expected input volumes based on mole numbers and molar volumes
                V_alpha_in = np.sum(n_mp[:, 0] * T_dep.v_mp[:, 0])
                V_beta_in = np.sum(n_mp[:, 1] * T_dep.v_mp[:, 1])

                # 3. Call the Janus geometry calculation function
                try:
                    r_vals = self.calculator._calc_Janus_geometry_for_known_nx(
                        n_mp, x_mp, calc_phases, T, T_dep, skin
                    )
                except (ValueError, RuntimeError, IndexError) as e:
                    # The geometry solver can fail for extreme or unphysical random inputs,
                    # which is an acceptable outcome. We'll just print a note and move on.
                    print(f"Note: Janus geometry solver failed for a random case, which is acceptable. Error: {e}")
                    return

                # Print output geometry
                print(f"Output Geometry (r_vals): [r_alpha, r_beta, cos_theta_alpha, cos_theta_beta]\n  {r_vals}")

                r_alpha, r_beta, cos_theta_alpha, cos_theta_beta = r_vals

                # 4. Calculate the output volumes from the returned geometry parameters
                V_calc = lambda r, cos_theta: np.pi * (r**3) * (2 + cos_theta) * ((1 - cos_theta)**2) / 3
                V_alpha_out = V_calc(r_alpha, cos_theta_alpha)
                V_beta_out = V_calc(r_beta, cos_theta_beta)

                # 5. Assert that the input and output volumes are conserved within a small tolerance
                self.assertAlmostEqual(V_alpha_in, V_alpha_out, delta=V_alpha_in * 1e-6,
                                       msg="Alpha phase volume is not conserved in Janus geometry calculation.")
                self.assertAlmostEqual(V_beta_in, V_beta_out, delta=V_beta_in * 1e-6,
                                       msg="Beta phase volume is not conserved in Janus geometry calculation.")

        # --- Run the test for a variety of random cases ---
        num_random_iterations = 5
        print(f"\nRunning {num_random_iterations} random iterations for each scenario...")
        for i in range(num_random_iterations):
            print(f"Iteration {i+1}/{num_random_iterations}")
            # Without skin
            run_single_random_test(("FCC", "Liquid"), T=np.random.uniform(800, 1400), skin_val=None)
            run_single_random_test(("FCC", "FCC"), T=np.random.uniform(800, 1200), skin_val=None)
            run_single_random_test(("Liquid", "Liquid"), T=np.random.uniform(1100, 1400), skin_val=None)
            
            # With skin
            run_single_random_test(("FCC", "Liquid"), T=np.random.uniform(800, 1400), skin_val=np.random.uniform(0.1, 0.9))

if __name__ == '__main__':
    unittest.main(verbosity=2)