"""
Integration tests for the Multi-Agent Debate System.

Tests the complete debate flow with mocked OpenAI responses.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Mock the openai module before importing debate_system
sys.modules['openai'] = MagicMock()

from debate_system import (
    DebateSystem,
    DebateConfig,
    DebateResult,
    Contribution,
    AuthenticationError,
    RoundType
)


class TestDebateSystemIntegration(unittest.TestCase):
    """Integration tests for complete debate flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "test-api-key-12345"
        self.test_problem = "Write a Python function to calculate factorial"
        
    def test_complete_debate_flow_with_three_agents_two_rounds(self):
        """Test complete debate with 3 agents and 2 rounds."""
        # Get the mocked openai module
        import openai
        
        # Mock OpenAI client and responses
        mock_client = MagicMock()
        openai.OpenAI.return_value = mock_client
        
        # Mock the models.list() call for API key validation
        mock_client.models.list.return_value = []
        
        # Mock chat completion responses
        mock_responses = [
            # Round 1 - Proposals
            "Proposal from agent 1: Use recursive approach",
            "Proposal from agent 2: Use iterative approach",
            "Proposal from agent 3: Use memoization approach",
            # Round 2 - Critiques
            "Critique from agent 2: Iterative is more efficient",
            "Critique from agent 3: Add error handling",
            "Critique from agent 1: Combine iterative with validation",
        ]
        
        mock_choice = Mock()
        mock_message = Mock()
        
        # Set up side_effect to return different responses
        response_iter = iter(mock_responses)
        def get_next_response(*args, **kwargs):
            mock_message.content = next(response_iter)
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            return mock_response
        
        mock_client.chat.completions.create.side_effect = get_next_response
        
        # Create debate system
        config = DebateConfig(
            api_key=self.test_api_key,
            num_agents=3,
            max_rounds=2,
            model="gpt-4"
        )
        
        debate_system = DebateSystem(config)
        
        # Run debate
        result = debate_system.start_debate(self.test_problem)
        
        # Verify result structure
        self.assertIsInstance(result, DebateResult)
        self.assertIsNotNone(result.final_solution)
        self.assertEqual(result.total_rounds, 2)
        self.assertEqual(len(result.contributing_agents), 3)
        
        # Verify debate history
        self.assertEqual(len(result.debate_history), 6)  # 3 agents * 2 rounds
        
        # Verify round 1 contributions are proposals
        round_1_contribs = [c for c in result.debate_history if c.round_number == 1]
        self.assertEqual(len(round_1_contribs), 3)
        for contrib in round_1_contribs:
            self.assertEqual(contrib.contribution_type, "proposal")
        
        # Verify round 2 contributions are critiques
        round_2_contribs = [c for c in result.debate_history if c.round_number == 2]
        self.assertEqual(len(round_2_contribs), 3)
        for contrib in round_2_contribs:
            self.assertEqual(contrib.contribution_type, "critique")
        
        # Verify final solution is from last round
        self.assertIn("agent", result.metadata["final_solution_agent"])
        self.assertEqual(result.metadata["final_solution_round"], 2)
        
    def test_debate_history_structure(self):
        """Test that debate history maintains proper structure."""
        # Get the mocked openai module
        import openai
        
        # Mock OpenAI client
        mock_client = MagicMock()
        openai.OpenAI.return_value = mock_client
        mock_client.models.list.return_value = []
        
        # Simple mock responses
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create debate system with 2 agents, 2 rounds
        config = DebateConfig(
            api_key=self.test_api_key,
            num_agents=2,
            max_rounds=2
        )
        
        debate_system = DebateSystem(config)
        result = debate_system.start_debate(self.test_problem)
        
        # Verify each contribution has required fields
        for contrib in result.debate_history:
            self.assertIsInstance(contrib, Contribution)
            self.assertIsNotNone(contrib.agent_id)
            self.assertIsNotNone(contrib.round_number)
            self.assertIsNotNone(contrib.contribution_type)
            self.assertIsNotNone(contrib.content)
            self.assertIsInstance(contrib.timestamp, datetime)
            
    def test_final_solution_selection(self):
        """Test that final solution is selected from last round."""
        # Get the mocked openai module
        import openai
        
        # Mock OpenAI client
        mock_client = MagicMock()
        openai.OpenAI.return_value = mock_client
        mock_client.models.list.return_value = []
        
        # Mock responses with identifiable content
        responses = [
            "Round 1 Agent 1",
            "Round 1 Agent 2",
            "Round 2 Agent 1 - Final",
            "Round 2 Agent 2",
        ]
        
        response_iter = iter(responses)
        def get_response(*args, **kwargs):
            content = next(response_iter)
            mock_message = Mock()
            mock_message.content = content
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            return mock_response
        
        mock_client.chat.completions.create.side_effect = get_response
        
        # Create debate system
        config = DebateConfig(
            api_key=self.test_api_key,
            num_agents=2,
            max_rounds=2
        )
        
        debate_system = DebateSystem(config)
        result = debate_system.start_debate(self.test_problem)
        
        # Verify final solution is from round 2
        self.assertEqual(result.metadata["final_solution_round"], 2)
        self.assertIn("Round 2", result.final_solution)


if __name__ == '__main__':
    unittest.main()
