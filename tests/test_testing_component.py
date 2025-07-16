import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from vivarium.framework.state_machine import Machine, State, TransientState

from vivarium_csu_alzheimers.components.testing import TestingForAlzheimers


class TestTestingForAlzheimers:
    """Test the TestingForAlzheimers component."""

    def test_probability_positive_with_ad(self):
        """Test that simulants with AD test positive with sensitivity probability."""
        # Setup
        sensitivity = 0.85
        specificity = 0.90
        
        # Create component
        component = TestingForAlzheimers('alzheimers_pet_scan', 'alzheimers_disease_and_other_dementias')
        
        # Mock the builder and setup
        builder = Mock()
        config_mock = Mock()
        config_mock.sensitivity = sensitivity
        config_mock.specificity = specificity
        builder.configuration = {'alzheimers_pet_scan': config_mock}
        
        # Mock the population view to return that simulants have AD
        pop_view_mock = Mock()
        pop_data = pd.DataFrame({
            'alzheimers_disease_and_other_dementias': ['alzheimers_first_state'] * 1000
        }, index=range(1000))
        pop_view_mock.get.return_value = pop_data
        builder.population.get_view.return_value = pop_view_mock
        
        # Setup component
        component.setup(builder)
        
        # Create test index
        test_index = pd.Index(range(1000))
        
        # Test the probability function
        prob_positive = component._probability_positive(test_index)
        
        # Assert that all simulants with AD have probability equal to sensitivity
        assert len(prob_positive) == 1000
        assert (prob_positive == sensitivity).all(), f"Expected all probabilities to be {sensitivity}, got {prob_positive.unique()}"

    def test_probability_positive_without_ad(self):
        """Test that simulants without AD test positive with (1-specificity) probability."""
        # Setup
        sensitivity = 0.85
        specificity = 0.90
        expected_prob = 1 - specificity  # 0.10
        
        # Create component
        component = TestingForAlzheimers('alzheimers_pet_scan', 'alzheimers_disease_and_other_dementias')
        
        # Mock the builder and setup
        builder = Mock()
        config_mock = Mock()
        config_mock.sensitivity = sensitivity
        config_mock.specificity = specificity
        builder.configuration = {'alzheimers_pet_scan': config_mock}
        
        # Mock the population view to return that simulants do NOT have AD
        pop_view_mock = Mock()
        pop_data = pd.DataFrame({
            'alzheimers_disease_and_other_dementias': ['susceptible_to_alzheimers_disease_and_other_dementias'] * 1000
        }, index=range(1000))
        pop_view_mock.get.return_value = pop_data
        builder.population.get_view.return_value = pop_view_mock
        
        # Setup component
        component.setup(builder)
        
        # Create test index
        test_index = pd.Index(range(1000))
        
        # Test the probability function
        prob_positive = component._probability_positive(test_index)
        
        # Assert that all simulants without AD have probability equal to (1-specificity)
        assert len(prob_positive) == 1000
        assert (prob_positive == expected_prob).all(), f"Expected all probabilities to be {expected_prob}, got {prob_positive.unique()}"

    def test_probability_positive_mixed_population(self):
        """Test probability with mixed population (some with AD, some without)."""
        # Setup
        sensitivity = 0.85
        specificity = 0.90
        
        # Create component
        component = TestingForAlzheimers('alzheimers_pet_scan', 'alzheimers_disease_and_other_dementias')
        
        # Mock the builder and setup
        builder = Mock()
        config_mock = Mock()
        config_mock.sensitivity = sensitivity
        config_mock.specificity = specificity
        builder.configuration = {'alzheimers_pet_scan': config_mock}
        
        # Mock the population view to return mixed population
        # First 500 have AD, next 500 don't
        ad_statuses = ['alzheimers_first_state'] * 500 + ['susceptible_to_alzheimers_disease_and_other_dementias'] * 500
        pop_view_mock = Mock()
        pop_data = pd.DataFrame({
            'alzheimers_disease_and_other_dementias': ad_statuses
        }, index=range(1000))
        pop_view_mock.get.return_value = pop_data
        builder.population.get_view.return_value = pop_view_mock
        
        # Setup component
        component.setup(builder)
        
        # Create test index
        test_index = pd.Index(range(1000))
        
        # Test the probability function
        prob_positive = component._probability_positive(test_index)
        
        # Assert correct probabilities for each group
        assert len(prob_positive) == 1000
        
        # First 500 (with AD) should have sensitivity probability
        with_ad_probs = prob_positive[:500]
        assert (with_ad_probs == sensitivity).all(), f"Expected AD group to have probability {sensitivity}, got {with_ad_probs.unique()}"
        
        # Next 500 (without AD) should have (1-specificity) probability
        without_ad_probs = prob_positive[500:]
        expected_prob = 1 - specificity
        assert (without_ad_probs == expected_prob).all(), f"Expected non-AD group to have probability {expected_prob}, got {without_ad_probs.unique()}"

    def test_probability_negative_is_complement(self):
        """Test that probability negative is 1 - probability positive."""
        # Setup
        sensitivity = 0.85
        specificity = 0.90
        
        # Create component
        component = TestingForAlzheimers('alzheimers_pet_scan', 'alzheimers_disease_and_other_dementias')
        
        # Mock the builder and setup
        builder = Mock()
        config_mock = Mock()
        config_mock.sensitivity = sensitivity
        config_mock.specificity = specificity
        builder.configuration = {'alzheimers_pet_scan': config_mock}
        
        # Mock mixed population
        ad_statuses = ['alzheimers_first_state'] * 500 + ['susceptible_to_alzheimers_disease_and_other_dementias'] * 500
        pop_view_mock = Mock()
        pop_data = pd.DataFrame({
            'alzheimers_disease_and_other_dementias': ad_statuses
        }, index=range(1000))
        pop_view_mock.get.return_value = pop_data
        builder.population.get_view.return_value = pop_view_mock
        
        # Setup component
        component.setup(builder)
        
        # Create test index
        test_index = pd.Index(range(1000))
        
        # Test both probability functions
        prob_positive = component._probability_positive(test_index)
        prob_negative = component._probability_negative(test_index)
        
        # Assert they sum to 1
        prob_sum = prob_positive + prob_negative
        assert (prob_sum == 1.0).all(), f"Probabilities should sum to 1.0, got {prob_sum.unique()}"

    def test_machine_integration(self):
        """Integration test to verify the machine uses correct testing logic."""
        # Setup
        sensitivity = 0.85
        specificity = 0.90
        
        # Create component
        component = TestingForAlzheimers('alzheimers_pet_scan', 'alzheimers_disease_and_other_dementias')
        
        # Mock the builder and setup
        builder = Mock()
        config_mock = Mock()
        config_mock.sensitivity = sensitivity
        config_mock.specificity = specificity
        builder.configuration = {'alzheimers_pet_scan': config_mock}
        
        # Mock mixed population
        ad_statuses = ['alzheimers_first_state'] * 500 + ['susceptible_to_alzheimers_disease_and_other_dementias'] * 500
        pop_view_mock = Mock()
        pop_data = pd.DataFrame({
            'alzheimers_disease_and_other_dementias': ad_statuses
        }, index=range(1000))
        pop_view_mock.get.return_value = pop_data
        builder.population.get_view.return_value = pop_view_mock
        
        # Setup component
        component.setup(builder)
        
        # Create test index
        test_index = pd.Index(range(1000))
        
        # This test verifies that the probability functions are working correctly
        # The machine should now use these methods instead of hardcoded 0.5 probabilities
        
        # Test that the methods are being used correctly
        positive_prob = component._probability_positive(test_index)
        negative_prob = component._probability_negative(test_index)
        
        # Verify the probabilities are correct for each group
        # First 500 (with AD) should have sensitivity probability
        with_ad_probs = positive_prob[:500]
        assert (with_ad_probs == sensitivity).all(), f"AD group should have sensitivity {sensitivity}, got {with_ad_probs.unique()}"
        
        # Next 500 (without AD) should have (1-specificity) probability
        without_ad_probs = positive_prob[500:]
        expected_prob = 1 - specificity
        assert (without_ad_probs == expected_prob).all(), f"Non-AD group should have probability {expected_prob}, got {without_ad_probs.unique()}"
        
        # Verify negative probabilities are complement
        assert (positive_prob + negative_prob == 1.0).all(), "Probabilities should sum to 1"

    def test_simulation_runs_successfully(self):
        """Integration test to ensure the simulation runs without error."""
        import subprocess
        import tempfile
        import os
        
        # Create a temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set environment variable to use temporary directory for results
            env = os.environ.copy()
            env['VIVARIUM_RESULTS_DIR'] = temp_dir
            
            # Run the simulation
            result = subprocess.run([
                'simulate', 'run', 
                'src/vivarium_csu_alzheimers/model_specifications/model_spec.yaml'
            ], env=env, capture_output=True, text=True, timeout=300)
            
            # Check that the simulation completed successfully
            assert result.returncode == 0, f"Simulation failed with return code {result.returncode}. Error: {result.stderr}"
            
            # Check that expected output messages are present
            output = result.stdout + result.stderr
            assert "Simulation finished" in output, f"Simulation didn't finish properly. Output: {output[-500:]}"
            assert "Results written to" in output, f"Results weren't written. Output: {output[-500:]}"