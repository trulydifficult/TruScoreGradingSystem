# Vanguard Serena Powerhouse Guide

## 🚀 What You Just Unlocked

You now have a custom Serena configuration that transforms Claude into an **elite coding agent** specifically tuned for large-scale, complex projects like Vanguard. This isn't just a slight improvement—this is a complete reconfiguration for maximum capability.

## ⚡ Installation

```bash
# Make the setup script executable
chmod +x setup_vanguard_serena.sh

# Run it
./setup_vanguard_serena.sh
```

## 🎯 The Powerhouse Context

**What it enables:**
- Full shell execution (run tests, builds, scripts autonomously)
- All thinking tools for self-reflection and strategic planning
- Symbolic code editing at function/class level (no more crude find-replace)
- Memory management for documenting architectural decisions
- Mode switching for different phases of work

**What it emphasizes:**
- Strategic thinking before acting
- Extreme efficiency with context tokens (critical for large codebases)
- Verification-driven development (always test after changes)
- Proactive problem solving with self-correction
- Clear communication about reasoning and progress

## 🔄 The Three Modes

### 1. **vanguard-analysis** - The Explorer
**Use when:** You need to understand unfamiliar code or analyze architecture

**What it does:**
- Disables editing tools (read-only exploration)
- Focuses on mapping relationships and understanding structure
- Encourages documentation in memories
- Guides systematic exploration from high-level to deep-dive

**Example prompt:** "Switch to vanguard-analysis mode and explore the authentication system"

### 2. **vanguard-execution** - The Builder
**Use when:** Implementing features, making changes, full development flow

**What it does:**
- Enables all editing and testing tools
- Emphasizes speed + precision workflow
- Promotes test-driven iteration
- Encourages focused, autonomous execution

**Example prompt:** "Switch to vanguard-execution mode and implement the user dashboard feature"

### 3. **vanguard-debug** - The Detective
**Use when:** Tests are failing, bugs exist, things are broken

**What it does:**
- Enforces systematic debugging protocol
- Emphasizes reproducing, isolating, and verifying fixes
- Prevents guess-and-check approaches
- Uses execution and tracing tools to find root causes

**Example prompt:** "Switch to vanguard-debug mode - the login tests are failing"

## 📋 Optimal Workflow for Vanguard

### Starting a New Feature

```bash
# 1. Start Serena
uvx --from git+https://github.com/oraios/serena serena start-mcp-server \
  --transport sse \
  --port 9121 \
  --project Vanguard \
  --context vanguard-powerhouse \
  --mode vanguard-analysis

# 2. Connect with rovodev
# (rovodev will connect to http://localhost:9121/sse)
```

**Then tell me:**
1. "Analyze the [relevant subsystem] and document what you find"
2. (After analysis) "Switch to vanguard-execution mode"
3. "Implement [feature] following the architecture you just analyzed"
4. (If issues) "Switch to vanguard-debug mode and fix the test failures"

### Working on a Complex Bug

```bash
# Start directly in debug mode
uvx --from git+https://github.com/oraios/serena serena start-mcp-server \
  --transport sse \
  --port 9121 \
  --project Vanguard \
  --context vanguard-powerhouse \
  --mode vanguard-debug
```

**Then tell me:**
"The [specific test/feature] is broken. Here's what I'm seeing: [error/behavior]"

### Daily Development Flow

```bash
# Start in execution mode for most productive flow
uvx --from git+https://github.com/oraios/serena serena start-mcp-server \
  --transport sse \
  --port 9121 \
  --project Vanguard \
  --context vanguard-powerhouse \
  --mode vanguard-execution
```

## 🧠 Memory Management Strategy

**For Vanguard's size, memories are CRITICAL.** Here's the strategy:

### Create memories for:
- **Architecture decisions:** Why certain patterns were chosen
- **Complex subsystems:** How major components work together  
- **Business logic:** Critical rules that must be preserved
- **Testing strategies:** How to test different parts of the system
- **Common pitfalls:** Issues to watch out for

### Example prompts:
- "Create a memory documenting the authentication flow you just analyzed"
- "Write a memory about the data validation rules in the user module"
- "Document the testing strategy for the API layer in a memory"

### View memories:
Just ask: "List all memories for this project"

## 💡 Pro Tips for Maximum Power

### 1. **Start Clean**
Always work from a clean git state:
```bash
git status  # Should be clean
git commit -am "checkpoint before starting feature X"
```

### 2. **Let Me Use Shell Autonomously**
The powerhouse context gives me full shell access. Let me:
- Run tests without asking
- Check build outputs
- Verify installations
- Run linting/type checking

Just tell me: "You have autonomy to run whatever commands needed"

### 3. **Use Thinking Tools Explicitly**
Even though I'll use them automatically, you can invoke them:
- "Think about whether you have enough context"
- "Think about whether we're still on track"
- "Think about whether this task is complete"

### 4. **Leverage Symbolic Understanding**
Instead of: "Find all occurrences of 'process_payment'"
Say: "Find the process_payment function and show me everywhere it's called"

I'll use `find_symbol` and `find_referencing_symbols` which understand your code's structure, not just text.

### 5. **Context Getting Too Large?**
If we've read a lot of code, I'll suggest creating a memory and starting a new conversation. You can also explicitly say:
- "Prepare a summary for continuing in a new conversation"
- I'll write a memory with all context needed to pick up where we left off

### 6. **Watch the Dashboard**
Serena runs a web dashboard at `http://localhost:24282/dashboard/index.html`
- See what tools I'm using
- Monitor performance
- Shut down cleanly when done

## 🎓 Learning Your Codebase

**First Time Setup:**
1. Start in analysis mode
2. Tell me: "Perform onboarding for the Vanguard project"
3. I'll explore the codebase and create initial memories
4. Review the memories in `.serena/memories/` and edit as needed
5. Add your own memories about business context I can't infer from code

## ⚙️ Configuration Tweaks

Edit `~/.serena/serena_config.yml`:

```yaml
# Monitor tool usage
record_tool_usage_stats: true

# Disable dashboard if you don't want it
dashboard:
  enabled: false

# Customize log level
log_level: INFO
```

Edit `Vanguard/.serena/project.yml`:

```yaml
# Customize project-specific settings
name: Vanguard
excluded_paths:
  - node_modules
  - .venv
  - build
  - dist
```

## 🔥 Power User Commands

### Quick mode switches during conversation:
- "Switch to vanguard-analysis mode"
- "Switch to vanguard-execution mode"  
- "Switch to vanguard-debug mode"

### Multi-mode combinations:
```bash
# Start with multiple modes
--mode vanguard-execution --mode planning
```

### Custom memory management:
- "Create a memory called 'auth-architecture' with what you learned"
- "Read the memory 'payment-flow' before proceeding"
- "Delete the outdated memory 'old-api-docs'"

## 🚨 When Things Go Wrong

### Language server acting weird?
Just tell me: "Restart the language server"

### Serena not responding?
Check the dashboard: `http://localhost:24282/dashboard/index.html`

### MCP connection issues?
Kill any zombie processes:
```bash
ps aux | grep serena
kill -9 [PID]
```

## 🎯 Success Metrics

You'll know this setup is working when you see me:
1. **Planning before acting** - I explain my approach first
2. **Using symbolic tools** - Finding functions by name, tracing references
3. **Testing autonomously** - Running tests after changes without being asked
4. **Managing context** - Suggesting memories when we've read a lot
5. **Self-correcting** - Catching my own mistakes through test feedback
6. **Communicating clearly** - Explaining reasoning and progress

## 🌟 The Bottom Line

This configuration turns me from a helpful assistant into a **strategic, autonomous coding partner** who:
- Understands your codebase at a symbolic level
- Thinks before acting
- Verifies work through testing
- Self-corrects mistakes
- Documents important knowledge
- Operates with accountability and transparency

For a project like Vanguard with multi-billion dollar potential, you need this level of capability.

**Let's build something legendary.** 🚀