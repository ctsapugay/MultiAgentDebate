# Multi-Agent Debate System

A lightweight Python framework for orchestrating collaborative problem-solving through structured debate between multiple OpenAI language model instances.

## Overview

The Multi-Agent Debate System enables multiple AI agents to collaboratively solve problems through a structured debate process. Each agent proposes solutions, critiques others' proposals, and refines ideas across multiple rounds, ultimately converging on an optimized solution.

## Features

- **Multiple Agent Orchestration**: Configure 2+ agents to participate in debates
- **Structured Rounds**: First round for proposals, subsequent rounds for critiques and refinements
- **Agent Rotation**: Prevents bias by rotating agent order each round
- **Complete History Tracking**: Records all contributions with timestamps and metadata
- **Flexible Configuration**: Customize number of agents, rounds, model, and temperature
- **CLI Interface**: Easy-to-use command-line interface with multiple output options

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ctsapugay/MultiAgentDebate.git
cd MultiAgentDebate
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your-actual-api-key-here
```

## Usage

### Basic Usage

```bash
python debate_system.py "Write a Python function to check if a number is prime"
```

### With Custom Configuration

```bash
# Specify number of agents and rounds
python debate_system.py "Implement a binary search algorithm" --agents 5 --rounds 4

# Use a different model
python debate_system.py "Create a REST API endpoint" --model gpt-3.5-turbo

# Adjust temperature for more creative responses
python debate_system.py "Design a caching strategy" --temperature 0.9
```

### Output Options

```bash
# Show only the debate history
python debate_system.py "Your problem" --history-only

# Show only the final solution
python debate_system.py "Your problem" --solution-only
```

### Command-Line Arguments

- `problem` (required): The problem statement or task description
- `--agents`: Number of agents (default: 3)
- `--rounds`: Maximum number of rounds (default: 3)
- `--model`: OpenAI model to use (default: gpt-4)
- `--temperature`: Temperature parameter 0-2 (default: 0.7)
- `--history-only`: Display only debate history
- `--solution-only`: Display only final solution

### Environment Variables

The system uses a `.env` file to manage configuration:

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## How It Works

1. **Initialization**: Creates the specified number of agent instances with unique identifiers
2. **Round 1 (Proposals)**: Each agent proposes an initial solution to the problem
3. **Subsequent Rounds (Critiques)**: Agents review previous contributions and provide critiques or refined solutions
4. **Solution Selection**: The system selects the final solution from the last round
5. **Output**: Displays complete debate history and final solution with metadata

## Example Output

```
ROUND 1 - PROPOSAL
────────────────────────────────────────────────────────────────────────────────
[agent_1]
Proposal from agent 1...

[agent_2]
Proposal from agent 2...

ROUND 2 - CRITIQUE
────────────────────────────────────────────────────────────────────────────────
[agent_1]
Critique and refinement...

FINAL SOLUTION
================================================================================
Selected from: agent_2
Round: 2
Total rounds: 2
Contributing agents: agent_1, agent_2, agent_3

[Final solution content...]
```

## Testing

Run the test suite:

```bash
python -m pytest test_debate_system.py -v
```

The tests use mocked OpenAI responses to verify:
- Complete debate flow with multiple agents and rounds
- Debate history structure and integrity
- Final solution selection logic

## Architecture

### Core Components

- **DebateSystem**: Main orchestrator managing the debate lifecycle
- **DebateConfig**: Configuration parameters for initialization
- **Contribution**: Individual agent proposals or critiques
- **DebateResult**: Final output containing solution and complete history

### Data Flow

```
Problem Input → Agent Initialization → Round 1 (Proposals) → 
Round N (Critiques) → Solution Selection → Final Result
```

## Requirements

- Python 3.7+
- OpenAI Python SDK (`openai`)
- python-dotenv (`python-dotenv`)
- Valid OpenAI API key

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built as a framework for exploring collaborative AI problem-solving through structured multi-agent debates.
