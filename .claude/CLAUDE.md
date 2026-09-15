# Claude Code Agent System for cisTEM

This directory contains specialized agents that assist with various development tasks in the cisTEM project. Agents are autonomous, task-focused AI assistants that Claude Code can invoke to handle complex, multi-step operations.

## What Are Agents?

Agents are specialized AI assistants designed for specific development workflows. When Claude Code encounters a task that matches an agent's expertise, it can delegate that work to the agent, which operates independently with access to appropriate tools and returns a comprehensive result.

## Available Agents

### Build & Compilation

#### **cpp-build-expert** (`agents/cpp-build-expert.md`)
**Purpose**: Compile C++ code and provide concise build diagnostics.

**When to use**:
- After modifying C++ source files
- When you need to verify compilation succeeds
- To diagnose build errors without verbose compiler output

**What it does**:
- Executes the build using `/build-cistem` command
- Filters template metaprogramming errors to root causes
- Combines file locations with error messages (`filename:line: error`)
- Provides actionable diagnostics for linker and compiler errors
- Returns clean SUCCESS/FAILED reports

**Example**: "I've updated the FFT wrapper class. Let's verify it compiles."

### Documentation

#### **doxygen-doc-expert** (`agents/doxygen-doc-expert.md`)
**Purpose**: Add high-value, LLM-friendly Doxygen documentation.

**When to use**:
- After writing new functions or classes
- When existing documentation is sparse or missing
- During API refactoring that changes behavior

**What it does**:
- Analyzes code context to determine documentation needs
- Adds Doxygen tags that maximize information density
- Focuses on non-obvious constraints, performance characteristics, edge cases
- Avoids documenting what's already clear from code
- Creates structured knowledge for AI code completion and navigation

**Example**: "This new particle picking algorithm needs proper documentation."

### Testing

#### **unit-test-architect** (`agents/unit-test-architect.md`)
**Purpose**: Create comprehensive unit tests for C++17/CUDA code.

**When to use**:
- After implementing new functionality
- When fixing bugs (to add regression tests)
- For code lacking test coverage
- When refactoring changes API contracts

**What it does**:
- Designs rigorous Catch2 v3 test suites
- Creates tests for edge cases, boundary conditions, negative paths
- Handles GPU-gated tests with CPU fallbacks
- Integrates with cisTEM's test infrastructure
- Provides realistic test data and fixtures

**Example**: "I've implemented a new binary protocol parser. We need comprehensive tests."

#### **gpu-test-debugger** (`agents/gpu-test-debugger.md`)
**Purpose**: Debug functional and console test failures with systematic GPU-aware investigation.

**When to use**:
- When `samples_functional_testing` or `console_test` fails
- For GPU memory corruption or race condition symptoms
- When tests produce incorrect numerical results
- For non-deterministic behavior in scientific computing

**What it does**:
- Establishes reproducible baselines with reference binaries
- Uses compute-sanitizer, cuda-gdb, and cisTEM debugging macros
- Systematically tests hypotheses (memory corruption, race conditions, numerical issues)
- Provides root cause analysis with verification steps
- Leverages `/build-cistem` to discover build directories automatically

**Example**: "The refine3d functional test is failing intermittently. Can you help debug it?"

### Version Control

#### **git-merge-expert** (`agents/git-merge-expert.md`)
**Purpose**: Systematic merge conflict resolution with phased categorization and VS Code integration.

**When to use**:
- When encountering merge conflicts during git merge operations
- For complex merges affecting multiple file categories
- When conflicts span structural, build system, implementation, and documentation changes

**What it does**:
- Categorizes conflicts into phases (structural, build, implementation, documentation)
- Integrates with VS Code merge editor for visual conflict resolution
- Provides clear recommendations for each conflict (Accept Ours/Theirs/Both/Manual)
- Interactive prompt pattern after opening each conflict file
- Stages files immediately after resolution
- Verifies build after merge completion
- Cleans up merge backup files

**Example**: "I'm merging the feature branch and have 9 conflicted files across different categories."

### Security

#### **red-team-security-tester** (`agents/red-team-security-tester.md`)
**Purpose**: Identify security vulnerabilities and attack surfaces.

**When to use**:
- After implementing network protocol parsers
- When adding IPC mechanisms or GPU code
- Before major releases
- During code review of security-critical components

**What it does**:
- Enumerates attack surfaces in network, IPC, and GPU code
- Identifies vulnerabilities (buffer overflows, race conditions, TOCTOU)
- Provides concrete, automatable proof-of-concept exploits
- Maps findings to CWE classifications
- Analyzes trust boundaries and privilege escalation paths

**Example**: "I've implemented shared memory IPC between GUI and workers. Check for security issues."

#### **blue-team-defender** (`agents/blue-team-defender.md`)
**Purpose**: Provide defensive mitigations and hardening strategies.

**When to use**:
- After receiving red-team security findings
- When hardening code before deployment
- To design defense-in-depth measures

**What it does**:
- Assesses exploitability and blast radius of vulnerabilities
- Provides complete, compilable code fixes following cisTEM standards
- Creates comprehensive test coverage (unit, property-based, fuzz)
- Delivers hardened build configurations (compiler flags, CUDA sanitizers, container security)
- Implements detection and telemetry for monitoring

**Example**: "The red team found buffer overflows in our socket parser. Need mitigation strategies."

#### **purple-team-lead** (`agents/purple-team-lead.md`)
**Purpose**: Coordinate adversarial review of plans through red/blue team cycles.

**When to use**:
- When you have a detailed plan document and want to stress-test it
- Before implementing major architectural changes
- To validate testing or deployment strategies

**What it does**:
- Validates that plans are sufficiently detailed for review
- Designs structured red team (attack/critique) and blue team (defense/improvement) cycles
- Provides checkpoints with findings and recommendations
- Determines when to continue or conclude review cycles

**Example**: "I've documented the new database schema in design-plan.md. Run purple team review."

## How to Use Agents

Agents are invoked automatically by Claude Code when tasks match their expertise. You can also explicitly request an agent:

```
"Use the cpp-build-expert agent to compile this."
"Invoke the red-team-security-tester to analyze this socket code."
"Run the purple-team-lead on my architecture plan."
```

## Agent Architecture

Each agent is defined in a markdown file with YAML frontmatter:
- `name`: Unique identifier for the agent
- `description`: When and how to use the agent
- `tools`: Tools the agent has access to (optional, defaults to all)
- `model`: AI model to use (typically "sonnet")
- `color`: Visual identifier for the agent

The file contains the complete system prompt that defines the agent's expertise, working process, output format, and quality standards.

## Creating New Agents

When creating agents for cisTEM:
1. **Define clear scope**: Each agent should have a specific, well-defined purpose
2. **Provide examples**: Include concrete usage examples in the description
3. **Set quality standards**: Define what constitutes good output for this agent
4. **Document tools**: Specify which tools the agent needs (or use defaults)
5. **Test thoroughly**: Ensure the agent produces valuable, actionable results

## Best Practices

- **Let agents work autonomously**: Agents are designed to complete complex tasks without step-by-step guidance
- **Provide context**: When invoking agents, give them the context they need (files, objectives, constraints)
- **Trust the output**: Agent results are generally reliable and well-formatted
- **Use specialized agents**: Don't use general-purpose assistance for tasks with specialized agents
- **Chain agents strategically**: For example, use cpp-build-expert after making changes, then unit-test-architect to add tests

## Integration with cisTEM Workflows

Agents are particularly valuable for:
- **Rapid iteration**: Build, test, fix cycles become more efficient
- **Code quality**: Documentation and testing agents ensure consistency
- **Security**: Red/blue team agents proactively identify and fix vulnerabilities
- **Knowledge transfer**: Agents document patterns and decisions for future developers

The agent system transforms Claude Code from a code assistant into a multi-agent development team, each member bringing specialized expertise to the cisTEM project.
