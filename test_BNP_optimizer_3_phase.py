import unittest
from unittest.mock import MagicMock, patch, ANY
import numpy as np

# Import the classes to be tested
from BNP_optimizer_3_phase import BNPOptimizer3Phase, OptimizationResult3Phase
from configurations_3_phase import ThreePhaseConfiguration

class TestBNPOptimizer3Phase(unittest.TestCase):
    def setUp(self):
        """Set up the test harness."""
        self.config = ThreePhaseConfiguration()
        self.config.materials = ("Ag", "Cu")
        
        # Patch the calculator class so we don't instantiate the real one
        self.patcher = patch('BNP_optimizer_3_phase.GibbsEnergyCalculator3Phase')
        self.MockCalcClass = self.patcher.start()
        self.mock_calc_instance = self.MockCalcClass.return_value
        
        # Instantiate optimizer (it will use the mock calculator)
        self.optimizer = BNPOptimizer3Phase(self.config)

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        """Test that the optimizer initializes correctly."""
        self.assertEqual(self.optimizer.config, self.config)
        self.assertIsNotNone(self.optimizer.calculator)
        self.MockCalcClass.assert_called_with(self.config)

    def test_find_minimum_energy_janus_simple(self):
        """
        Test simple minimization without constraints (Janus).
        We mock the energy function to be a simple paraboloid: (x-0.3)^2 + (y-0.7)^2.
        """
        # Mock objective function behavior
        # Minimize: (A_ratio - 0.3)^2 + (B_ratio - 0.7)^2
        def side_effect_energy(A_ratio_alpha, B_ratio_alpha, **kwargs):
            return (A_ratio_alpha - 0.3)**2 + (B_ratio_alpha - 0.7)**2
        
        self.mock_calc_instance.calculate_total_energy.side_effect = side_effect_energy

        # Mock the final detailed calculation call
        self.mock_calc_instance._update_phases_based_on_skin.return_value = (('A', 'B'), MagicMock(exists=False))
        self.mock_calc_instance._get_T_dependent_vars.return_value = MagicMock()
        # Return dummy values for n_mp, x_mp, r_vals
        self.mock_calc_instance._calc_mole_splits_and_geo.return_value = (
            np.zeros((2, 2)), np.zeros((2, 2)), np.array([1.0, 1.0, 0.5])
        )

        result = self.optimizer.find_minimum_energy(
            T=1000, n_total=1.0, xB_total=0.5,
            primary_phases=("FCC", "Liquid"),
            geometry_type="Janus",
            has_skin=False
        )

        # Check if optimizer found the minimum at (0.3, 0.7)
        self.assertAlmostEqual(result.A_ratio_alpha, 0.3, places=2)
        self.assertAlmostEqual(result.B_ratio_alpha, 0.7, places=2)
        self.assertEqual(result.geometry_type, "Janus")
        self.assertEqual(result.G_min, 0.0) # Minimum value of our parabola is 0

    def test_find_minimum_energy_calculator_failure(self):
        """Test handling when calculator raises errors (simulating failure for all points)."""
        self.mock_calc_instance.calculate_total_energy.side_effect = ValueError("Simulated Calc Failure")
        
        result = self.optimizer.find_minimum_energy(
            T=1000, n_total=1.0, xB_total=0.5,
            primary_phases=("FCC", "Liquid"),
            geometry_type="Janus",
            has_skin=False
        )

        self.assertEqual(result.G_min, float('inf'))
        self.assertTrue(np.isnan(result.A_ratio_alpha))

    def test_core_shell_constraint_logic(self):
        """
        Test that Core-Shell geometry applies constraints.
        We simulate a scenario where high Alpha ratio creates a shell that is too thin.
        """
        # Setup Material Data for radius lookup
        matA = MagicMock(); matA.atomic_radius = 1.0
        matB = MagicMock(); matB.atomic_radius = 1.0
        self.mock_calc_instance.material_data = {"Ag": matA, "Cu": matB}

        # Mock `_calc_mole_splits_and_geo` to simulate shell thickness based on A_ratio
        def mock_calc_splits(A_ratio, B_ratio, *args, **kwargs):
            # Continuous thickness function to help SLSQP gradients:
            # Thickness = 10 * (1.0 - A_ratio). 
            # At A=0.8, Thick=2.0 (Limit t_min=2.0). A > 0.8 -> Violation.
            r_core = 10.0
            thickness = max(0.0, 10.0 * (1.0 - float(A_ratio)))
            r_shell = r_core + thickness
            
            x_mp = np.zeros((2, 2))
            x_mp[1, 1] = 0.5 # xB_shell = 0.5
            n_mp = np.zeros((2, 2))
            
            return n_mp, x_mp, np.array([r_core, r_shell])

        self.mock_calc_instance._calc_mole_splits_and_geo.side_effect = mock_calc_splits
        self.mock_calc_instance._update_phases_based_on_skin.return_value = (('A', 'B'), MagicMock(exists=False))
        self.mock_calc_instance._get_T_dependent_vars.return_value = MagicMock()

        # Mock Energy: Lower energy at higher A_ratio (forcing optimizer to push against constraint)
        # E = -A_ratio
        self.mock_calc_instance.calculate_total_energy.side_effect = lambda A_ratio_alpha, B_ratio_alpha, **kwargs: -A_ratio_alpha

        result = self.optimizer.find_minimum_energy(
            T=1000, n_total=1.0, xB_total=0.5,
            primary_phases=("FCC", "Liquid"),
            geometry_type="Core Shell",
            has_skin=False
        )

        # Without constraint, optimizer would go to 1.0.
        # With constraint (cutoff at >0.8), it should stop around 0.8.
        self.assertLessEqual(result.A_ratio_alpha, 0.85)
        self.assertNotEqual(result.G_min, float('inf'))
        
        # Verify that `_calc_mole_splits_and_geo` was called (implies constraint check ran)
        self.assertTrue(self.mock_calc_instance._calc_mole_splits_and_geo.called)

if __name__ == '__main__':
    unittest.main()
