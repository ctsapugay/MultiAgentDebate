"""
Multi-Agent Debate System

A lightweight framework for orchestrating collaborative problem-solving through
structured debate between multiple OpenAI language model instances.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class RoundType(Enum):
    """Type of debate round"""
    PROPOSAL = "proposal"
    CRITIQUE = "critique"


@dataclass
class DebateConfig:
    """Configuration parameters for debate initialization.
    
    Attributes:
        api_key: OpenAI API key for authentication
        num_agents: Number of agent instances to create (default: 3)
        max_rounds: Maximum number of debate rounds (default: 3)
        model: OpenAI model to use (default: "gpt-4")
        temperature: Temperature parameter for model responses (default: 0.7)
    """
    api_key: str
    num_agents: int = 3
    max_rounds: int = 3
    model: str = "gpt-4"
    temperature: float = 0.7


@dataclass
class Contribution:
    """Represents a single agent contribution (proposal or critique).
    
    Attributes:
        agent_id: Unique identifier for the contributing agent
        round_number: Round number when contribution was made
        contribution_type: Type of contribution ("proposal" or "critique")
        content: The actual content of the contribution
        timestamp: When the contribution was made
    """
    agent_id: str
    round_number: int
    contribution_type: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DebateContext:
    """Context information provided to agents for generating responses.
    
    Attributes:
        problem: The problem statement or task description
        round_number: Current round number
        previous_contributions: List of all previous contributions
        round_type: Type of current round ("proposal" or "critique")
    """
    problem: str
    round_number: int
    previous_contributions: List[Contribution]
    round_type: str


@dataclass
class Solution:
    """Represents the selected final solution.
    
    Attributes:
        content: The solution content
        agent_id: ID of the agent who contributed this solution
        round_number: Round number when solution was proposed
    """
    content: str
    agent_id: str
    round_number: int


@dataclass
class DebateResult:
    """Final output from the debate system.
    
    Attributes:
        final_solution: The selected final solution content
        contributing_agents: List of agent IDs that contributed to the solution
        total_rounds: Total number of rounds executed
        debate_history: Complete chronological list of all contributions
        metadata: Additional information about the debate process
    """
    final_solution: str
    contributing_agents: List[str]
    total_rounds: int
    debate_history: List[Contribution]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebateSystem:
    """Main orchestrator that coordinates the entire debate process.
    
    The DebateSystem manages the lifecycle of a multi-agent debate, including
    agent initialization, round progression, and solution selection.
    """
    
    # Prompt templates for different round types
    PROPOSAL_PROMPT = """You are participating in a collaborative debate to solve the following problem:

{problem}

This is round {round_number} of the debate. Please provide your initial solution proposal.

Be specific and provide concrete implementation details."""
    
    CRITIQUE_PROMPT = """You are participating in a collaborative debate to solve the following problem:

{problem}

Previous proposals and critiques from other agents:
{previous_contributions}

Please critique the existing proposals and suggest improvements or provide a refined solution.

Focus on identifying weaknesses and proposing concrete improvements."""
    
    def __init__(self, config: DebateConfig):
        """Initialize debate system with configuration.
        
        Args:
            config: DebateConfig object containing API key and debate parameters
            
        Raises:
            ValueError: If configuration is invalid
            AuthenticationError: If OpenAI API key is invalid
        """
        self._validate_config(config)
        self.config = config
        self.agent_ids: List[str] = []
        self.debate_history: List[Contribution] = []
        self.current_round = 0
        
        # Import OpenAI here to validate API key early
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.api_key)
            # Validate API key by making a simple call
            self._validate_api_key()
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise AuthenticationError(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Create agent instances (stored as list of IDs)
        self._create_agents()
    
    def _validate_config(self, config: DebateConfig):
        """Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        if config.num_agents < 2:
            raise ValueError("Number of agents must be at least 2")
        if config.max_rounds < 1:
            raise ValueError("Maximum rounds must be at least 1")
        if not config.api_key:
            raise ValueError("API key is required")
        if config.temperature < 0 or config.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
    
    def _validate_api_key(self):
        """Validate OpenAI API key by making a test call.
        
        Raises:
            AuthenticationError: If API key is invalid
        """
        try:
            # Make a minimal API call to validate the key
            self.client.models.list()
        except Exception as e:
            raise AuthenticationError(f"Invalid OpenAI API key: {str(e)}")
    
    def _create_agents(self):
        """Create multiple agent instances (stored as list of IDs).
        
        Each agent is assigned a unique identifier for tracking contributions.
        """
        self.agent_ids = [f"agent_{i+1}" for i in range(self.config.num_agents)]
    
    def start_debate(self, problem: str) -> DebateResult:
        """Execute complete debate process and return final solution.
        
        This method orchestrates the full debate flow:
        1. First round: all agents propose initial solutions
        2. Subsequent rounds: agents critique and refine proposals
        3. Final round: select best solution
        
        Args:
            problem: The problem statement or task description
            
        Returns:
            DebateResult containing final solution and debate history
        """
        if not problem or not problem.strip():
            raise ValueError("Problem statement cannot be empty")
        
        # Reset state for new debate
        self.debate_history = []
        self.current_round = 0
        
        # Run debate rounds
        for round_num in range(1, self.config.max_rounds + 1):
            self.current_round = round_num
            
            # First round is proposals, subsequent rounds are critiques
            round_type = RoundType.PROPOSAL if round_num == 1 else RoundType.CRITIQUE
            
            self._run_round(problem, round_num, round_type)
        
        # Select final solution
        final_solution = self._select_final_solution()
        
        # Gather contributing agents
        contributing_agents = list(set(c.agent_id for c in self.debate_history))
        
        return DebateResult(
            final_solution=final_solution.content,
            contributing_agents=contributing_agents,
            total_rounds=self.current_round,
            debate_history=self.debate_history.copy(),
            metadata={
                "model": self.config.model,
                "num_agents": self.config.num_agents,
                "final_solution_agent": final_solution.agent_id,
                "final_solution_round": final_solution.round_number
            }
        )
    
    def _run_round(self, problem: str, round_number: int, round_type: RoundType):
        """Execute a single debate round.
        
        Args:
            problem: The problem statement
            round_number: Current round number
            round_type: Type of round (PROPOSAL or CRITIQUE)
        """
        # Prepare prompt based on round type
        if round_type == RoundType.PROPOSAL:
            prompt = self.PROPOSAL_PROMPT.format(
                problem=problem,
                round_number=round_number
            )
        else:
            # Format previous contributions for context
            previous_text = self._format_previous_contributions()
            prompt = self.CRITIQUE_PROMPT.format(
                problem=problem,
                previous_contributions=previous_text,
                round_number=round_number
            )
        
        # Rotate agent order to prevent bias (Requirement 2.3)
        # Rotate by round number so each round has a different starting agent
        rotation_offset = (round_number - 1) % len(self.agent_ids)
        rotated_agents = self.agent_ids[rotation_offset:] + self.agent_ids[:rotation_offset]
        
        # Collect responses from all agents
        for agent_id in rotated_agents:
            try:
                response = self._call_openai(prompt)
                
                contribution = Contribution(
                    agent_id=agent_id,
                    round_number=round_number,
                    contribution_type=round_type.value,
                    content=response,
                    timestamp=datetime.now()
                )
                
                self.debate_history.append(contribution)
                
            except Exception as e:
                # Log failure but continue with other agents
                print(f"Warning: Agent {agent_id} failed in round {round_number}: {str(e)}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant participating in a collaborative debate."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature
        )
        
        return response.choices[0].message.content
    
    def _format_previous_contributions(self) -> str:
        """Format previous contributions as context string for agents.
        
        Returns:
            Formatted string of previous contributions
        """
        if not self.debate_history:
            return "No previous contributions yet."
        
        formatted = []
        for contrib in self.debate_history:
            formatted.append(
                f"[{contrib.agent_id} - Round {contrib.round_number} - {contrib.contribution_type}]\n"
                f"{contrib.content}\n"
            )
        
        return "\n".join(formatted)
    
    def _select_final_solution(self) -> Solution:
        """Analyze contributions and select best solution.
        
        For the baseline implementation, we select the most recent proposal
        from the last round as it represents the most refined solution.
        
        Returns:
            Solution object containing the selected final solution
        """
        if not self.debate_history:
            raise ValueError("No contributions in debate history")
        
        # Get contributions from the last round
        last_round = max(c.round_number for c in self.debate_history)
        last_round_contributions = [
            c for c in self.debate_history 
            if c.round_number == last_round
        ]
        
        # Select the first contribution from the last round
        # (in future iterations, this could use more sophisticated selection)
        selected = last_round_contributions[0] if last_round_contributions else self.debate_history[-1]
        
        return Solution(
            content=selected.content,
            agent_id=selected.agent_id,
            round_number=selected.round_number
        )
    
    def get_debate_history(self) -> List[Contribution]:
        """Retrieve complete debate history.
        
        Returns:
            List of all contributions in chronological order
        """
        return self.debate_history.copy()


class AuthenticationError(Exception):
    """Raised when OpenAI API authentication fails."""
    pass


def format_debate_history(history: List[Contribution]) -> str:
    """Format debate history in a human-readable format.
    
    Args:
        history: List of contributions to format
        
    Returns:
        Formatted string representation of the debate history
    """
    if not history:
        return "No debate history available."
    
    output = []
    output.append("=" * 80)
    output.append("DEBATE HISTORY")
    output.append("=" * 80)
    
    current_round = 0
    for contrib in history:
        # Add round separator when round changes
        if contrib.round_number != current_round:
            current_round = contrib.round_number
            output.append(f"\n{'─' * 80}")
            output.append(f"ROUND {current_round} - {contrib.contribution_type.upper()}")
            output.append(f"{'─' * 80}\n")
        
        output.append(f"[{contrib.agent_id}]")
        output.append(contrib.content)
        output.append("")  # Empty line between contributions
    
    return "\n".join(output)


def format_final_solution(result: DebateResult) -> str:
    """Format the final solution in a human-readable format.
    
    Args:
        result: DebateResult containing the final solution
        
    Returns:
        Formatted string representation of the final solution
    """
    output = []
    output.append("\n" + "=" * 80)
    output.append("FINAL SOLUTION")
    output.append("=" * 80)
    output.append(f"\nSelected from: {result.metadata.get('final_solution_agent', 'unknown')}")
    output.append(f"Round: {result.metadata.get('final_solution_round', 'unknown')}")
    output.append(f"Total rounds: {result.total_rounds}")
    output.append(f"Contributing agents: {', '.join(result.contributing_agents)}")
    output.append(f"\n{result.final_solution}")
    output.append("\n" + "=" * 80)
    
    return "\n".join(output)


def main():
    """Main CLI function for the Multi-Agent Debate System.
    
    Example usage:
        # Basic usage (API key from .env file):
        python debate_system.py "Write a Python function to calculate fibonacci numbers"
        
        # With custom configuration:
        python debate_system.py "Implement a binary search algorithm" --agents 5 --rounds 4
        
        # Using a different model:
        python debate_system.py "Create a REST API endpoint" --model gpt-3.5-turbo
    
    The system will:
    1. Load API key from .env file
    2. Initialize the specified number of agents
    3. Run the debate for the specified number of rounds
    4. Display the complete debate history
    5. Show the final selected solution
    """
    import argparse
    import os
    import sys
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed. Install with: pip install python-dotenv", file=sys.stderr)
        print("Falling back to system environment variables.", file=sys.stderr)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate System for collaborative problem-solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Write a function to sort a list"
  %(prog)s "Implement a cache decorator" --agents 5 --rounds 4
  %(prog)s "Create a binary tree class" --model gpt-3.5-turbo
        """
    )
    
    parser.add_argument(
        "problem",
        type=str,
        help="The problem statement or task description for the agents to solve"
    )
    
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of agents to participate in the debate (default: 3)"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Maximum number of debate rounds (default: 3)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="OpenAI model to use (default: gpt-4)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature parameter for model responses (default: 0.7)"
    )
    
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Only display debate history, not the final solution"
    )
    
    parser.add_argument(
        "--solution-only",
        action="store_true",
        help="Only display final solution, not the debate history"
    )
    
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OpenAI API key is required.", file=sys.stderr)
        print("Please set OPENAI_API_KEY in your .env file or environment variables.", file=sys.stderr)
        print("See .env.example for reference.", file=sys.stderr)
        sys.exit(1)
    
    # Validate problem statement
    if not args.problem.strip():
        print("Error: Problem statement cannot be empty.", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create configuration
        config = DebateConfig(
            api_key=api_key,
            num_agents=args.agents,
            max_rounds=args.rounds,
            model=args.model,
            temperature=args.temperature
        )
        
        print(f"Initializing debate system with {args.agents} agents using {args.model}...")
        print(f"Problem: {args.problem}\n")
        
        # Initialize debate system
        debate_system = DebateSystem(config)
        
        print(f"Starting debate with {args.rounds} rounds...\n")
        
        # Run the debate
        result = debate_system.start_debate(args.problem)
        
        # Display results based on flags
        if not args.solution_only:
            print(format_debate_history(result.debate_history))
        
        if not args.history_only:
            print(format_final_solution(result))
        
        print("\nDebate completed successfully!")
        
    except AuthenticationError as e:
        print(f"Authentication Error: {e}", file=sys.stderr)
        print("\nPlease verify your OpenAI API key is valid.", file=sys.stderr)
        sys.exit(1)
        
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\nDebate interrupted by user.", file=sys.stderr)
        sys.exit(130)
        
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
