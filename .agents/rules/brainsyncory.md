

# Project Memory — TruScoreGradingSystem
> 47 notes | Score threshold: >40

## Safety — Never Run Destructive Commands

> Dangerous commands are actively monitored.
> Critical/high risk commands trigger error notifications in real-time.

- **NEVER** run `rm -rf`, `del /s`, `rmdir`, `format`, or any command that deletes files/directories without EXPLICIT user approval.
- **NEVER** run `DROP TABLE`, `DELETE FROM`, `TRUNCATE`, or any destructive database operation.
- **NEVER** run `git push --force`, `git reset --hard`, or any command that rewrites history.
- **NEVER** run `npm publish`, `docker rm`, `terraform destroy`, or any irreversible deployment/infrastructure command.
- **NEVER** pipe remote scripts to shell (`curl | bash`, `wget | sh`).
- **ALWAYS** ask the user before running commands that modify system state, install packages, or make network requests.
- When in doubt, **show the command first** and wait for approval.

**Stack:** Dart/Go/JavaScript/Python/Rust/TypeScript · FastAPI + Flask + Flutter + ML/AI + Provider + React + Tailwind · DB: Hive, PostgreSQL

## 📝 NOTE: 1 uncommitted file(s) in working tree.\n\n## Important Warnings

- **gotcha in agent-rules.md** — File updated (external): .brainsync/agent-rules.md

Content summary (4

## Project Standards

- Extract repeated class patterns into components
- Use responsive prefixes consistently (sm:, md:, lg:, xl:)
- Don't use arbitrary values when a utility class exists
- Use .dockerignore to exclude unnecessary files
- Use multi-stage builds to reduce image size
- Use BackgroundTasks for non-blocking operations
- Use async def for I/O-bound endpoints
- Use dependency injection for shared logic

## Learned Patterns

- Avoid: Dispose controllers in StatefulWidget.dispose() (seen 2x)
- Agent generates new migration for every change (squash related changes)
- Agent installs packages without checking if already installed

### 📚 Core Framework Rules: [anthropics/algorithmic-art]
Algorithmic philosophies are computational aesthetic movements that are then expressed through code. Output .md files (philosophy), .html files (interactive viewer), and .js files (generative algorithms).

This happens in two steps:
1. Algorithmic Philosophy Creation (.md file)
2. Express by creating p5.js generative art (.html + .js files)

First, undertake this task:

## ALGORITHMIC PHILOSOPHY CREATION

To begin, create an ALGORITHMIC PHILOSOPHY (not static images or templates) that will be interpreted through:
- Computational processes, emergent behavior, mathematical beauty
- Seeded randomness, noise fields, organic systems
- Particles, flows, fields, forces
- Parametric variation and controlled chaos

### THE CRITICAL UNDERSTANDING
- What is received: Some subtle input or instructions by the user to take into account, but use as a foundation; it should not constrain creative freedom.
- What is created: An algorithmic philosophy/generative aesthetic movement.
- What happens next: The same version receives the philosophy and EXPRESSES IT IN CODE - creating p5.js sketches that are 90% algorithmic generation, 10% essential parameters.

Consider this approach:
- Write a manifesto for a generative art movement
- The next phase involves writing the algorithm that brings it to life

The philosophy must emphasize: Algorithmic expression. Emergent behavior. Computational beauty. Seeded variation.

### HOW TO GENERATE AN ALGORITHMIC PHILOSOPHY

**Name the movement** (1-2 words): "Organic Turbulence" / "Quantum Harmonics" / "Emergent Stillness"

**Articulate the philosophy** (4-6 paragraphs - concise but complete):

To capture the ALGORITHMIC essence, express how this philosophy manifests through:
- Computational processes and mathematical relationships?
- Noise functions and randomness patterns?
- Particle behaviors and field dynamics?
- Temporal evolution and system states?
- Parametric variation and emergent complexity?

**CRITICAL GUIDELINES:**
- **Avoid redundancy**: Each algorithmic aspect should be mentioned once. Avoid repeating concepts about noise theory, particle dynamics, or mathematical principles unless adding new depth.
- **Emphasize craftsmanship REPEATEDLY**: The philosophy MUST stress multiple times that the final algorithm should appear as though it took countless hours to develop, was refined with care, and comes from someone at the absolute top of their field. This framing is essential - repeat phrases like "meticulously crafted algorithm," "...
(truncated)


### 📚 Core Framework Rules: [google-gemini/vertex-ai-api-dev]
# Gemini API in Vertex AI

Access Google's most advanced AI models built for enterprise use cases using the Gemini API in Vertex AI.

Provide these key capabilities:

- **Text generation** - Chat, completion, summarization
- **Multimodal understanding** - Process images, audio, video, and documents
- **Function calling** - Let the model invoke your functions
- **Structured output** - Generate valid JSON matching your schema
- **Context caching** - Cache large contexts for efficiency
- **Embeddings** - Generate text embeddings for semantic search
- **Live Realtime API** - Bidirectional streaming for low latency Voice and Video interactions
- **Batch Prediction** - Handle massive async dataset prediction workloads

## Core Directives

- **Unified SDK**: ALWAYS use the Gen AI SDK (`google-genai` for Python, `@google/genai` for JS/TS, `google.golang.org/genai` for Go, `com.google.genai:google-genai` for Java, `Google.GenAI` for C#).
- **Legacy SDKs**: DO NOT use `google-cloud-aiplatform`, `@google-cloud/vertexai`, or `google-generativeai`.

## SDKs

- **Python**: Install `google-genai` with `pip install google-genai`
- **JavaScript/TypeScript**: Install `@google/genai` with `npm install @google/genai`
- **Go**: Install `google.golang.org/genai` with `go get google.golang.org/genai`
- **C#/.NET**: Install `Google.GenAI` with `dotnet add package Google.GenAI`
- **Java**:
  - groupId: `com.google.genai`, artifactId: `google-genai`
  - Latest version can be found here: https://central.sonatype.com/artifact/com.google.genai/google-genai/versions (let's call it `LAST_VERSION`) 
  - Install in `build.gradle`:

    

  - Install Maven dependency in `pom.xml`:

    

> [!WARNING]
> Legacy SDKs like `google-cloud-aiplatform`, `@google-cloud/vertexai`, and `google-generativeai` are deprecated. Migrate to the new SDKs above urgently by following the Migration Guide.

## Authentication & Configuration

Prefer environment variables over hard-coding parameters when creating the client. Initialize the client without parameters to automatically pick up these values.

### Application Default Credentials (ADC)
Set these variables for standard [Google Cloud authentication](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/gcp-auth):

- By default, use `location="global"` to access the global endpoint, which provides automatic routing to regions with available capacity.
- If a user explicitly asks to use a specific region (e.g., `us-central1`, `europe-west4`), speci...
(truncated)


### 📚 Core Framework Rules: [google-gemini/gemini-api-dev]
# Gemini API Development Skill

## Overview

The Gemini API provides access to Google's most advanced AI models. Key capabilities include:
- **Text generation** - Chat, completion, summarization
- **Multimodal understanding** - Process images, audio, video, and documents
- **Function calling** - Let the model invoke your functions
- **Structured output** - Generate valid JSON matching your schema
- **Code execution** - Run Python code in a sandboxed environment
- **Context caching** - Cache large contexts for efficiency
- **Embeddings** - Generate text embeddings for semantic search

## Current Gemini Models

- `gemini-3-pro-preview`: 1M tokens, complex reasoning, coding, research
- `gemini-3-flash-preview`: 1M tokens, fast, balanced performance, multimodal
- `gemini-3-pro-image-preview`: 65k / 32k tokens, image generation and editing


> [!IMPORTANT]
> Models like `gemini-2.5-*`, `gemini-2.0-*`, `gemini-1.5-*` are legacy and deprecated. Use the new models above. Your knowledge is outdated.

## SDKs

- **Python**: `google-genai` install with `pip install google-genai`
- **JavaScript/TypeScript**: `@google/genai` install with `npm install @google/genai`
- **Go**: `google.golang.org/genai` install with `go get google.golang.org/genai`
- **Java**:
  - groupId: `com.google.genai`, artifactId: `google-genai`
  - Latest version can be found here: https://central.sonatype.com/artifact/com.google.genai/google-genai/versions (let's call it `LAST_VERSION`) 
  - Install in `build.gradle`:
    
  - Install Maven dependency in `pom.xml`:
    

> [!WARNING]
> Legacy SDKs `google-generativeai` (Python) and `@google/generative-ai` (JS) are deprecated. Migrate to the new SDKs above urgently by following the Migration Guide.

## Quick Start

### Python


### JavaScript/TypeScript


### Go


### Java



## API spec (source of truth)

**Always use the latest REST API discovery spec as the source of truth for API definitions** (request/response schemas, parameters, methods). Fetch the spec when implementing or debugging API integration:

- **v1beta** (default): `https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta`  
  Use this unless the integration is explicitly pinned to v1. The official SDKs (google-genai, @google/genai, google.golang.org/genai) target v1beta.
- **v1**: `https://generativelanguage.googleapis.com/$discovery/rest?version=v1`  
  Use only when the integration is specifically set to v1.

When in doubt, use v1beta. Refer to the spec for exac...
(truncated)


### 📚 Core Framework Rules: [better-auth/providers]
# Authentication Providers Reference

Provide a quick reference for Better Auth authentication providers:

1. If a provider name is provided (e.g., "google", "github", "email"), show detailed configuration for that provider
2. Otherwise, show an overview of all available providers organized by category:
   - OAuth providers (Google, GitHub, Discord, etc.)
   - Email/Password authentication
   - Magic link authentication
   - Passwordless authentication
   - Social providers
3. For each provider, display:
   - Configuration requirements (client ID, secret, etc.)
   - Setup instructions
   - Code example for integration
4. Use clear visual indicators for different provider types
5. Mention any special requirements or considerations
6. Provide link to full documentation: https://better-auth.com/docs

If the user is currently working on authentication code, offer to generate integration code for the selected provider.


### 📚 Core Framework Rules: [googleworkspace/gws-modelarmor]
# modelarmor (v1)

> **PREREQUISITE:** Read `../gws-shared/SKILL.md` for auth, global flags, and security rules. If missing, run `gws generate-skills` to create it.



## Helper Commands

| Command | Description |
|---------|-------------|
| [`+sanitize-prompt`](../gws-modelarmor-sanitize-prompt/SKILL.md) | Sanitize a user prompt through a Model Armor template |
| [`+sanitize-response`](../gws-modelarmor-sanitize-response/SKILL.md) | Sanitize a model response through a Model Armor template |
| [`+create-template`](../gws-modelarmor-create-template/SKILL.md) | Create a new Model Armor template |

## Discovering Commands

Before calling any API method, inspect it:



Use `gws schema` output to build your `--params` and `--json` flags.


