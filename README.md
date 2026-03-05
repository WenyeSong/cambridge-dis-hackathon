# Agentic FactTrace: Multi-Agent Reasoning for Claim Faithfulness

This project was developed for the **FactTrace Hackathon @ University of Cambridge**, which challenges participants to build a **jury of AI agents** that debate whether a public claim faithfully represents an underlying fact.

## Problem

Each case contains two statements:

- **Internal Fact** – the original source truth  
- **External Claim** – a derived statement (e.g., headline or summary)

The task is to determine whether the claim **faithfully represents the fact** or whether its meaning has **mutated** through exaggeration, missing context, or framing.

## Approach

Instead of relying on a single LLM response, this system uses a **multi-agent reasoning pipeline** where different agents analyze the claim–fact relationship from distinct perspectives.

Agents include:

- **Grounding Agent** – extracts atomic propositions from both statements  
- **Forward Entailment Agent** – tests whether the fact logically implies the claim  
- **Reverse Entailment Agent** – tests whether the claim implies the fact  
- **Optimist Agent** – constructs the strongest charitable interpretation  
- **Pessimist Agent** – constructs the strongest adversarial interpretation  
- **Synthesizer Agent** – summarizes competing interpretations and explains the source of disagreement

This design allows the system to expose **multiple interpretations and reasoning paths**, rather than producing a single opaque verdict.

