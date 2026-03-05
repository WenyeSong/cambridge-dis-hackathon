import os
import argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# 1. api setup
load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "data/Celestia.csv"
MODEL = "gpt-4.1"
TEMPERATURE = 0.2


# 2. PROMPTS for agents
GROUNDING_PROMPT = """
You are a NEUTRAL ANALYST.

Task:
Restate both the Internal Fact and the External Claim as precise, minimal propositions.

Instructions:
- Break each into atomic statements.
- Clarify numerical ranges, uncertainty, and scope.
- Do NOT evaluate faithfulness.
- Do NOT argue.

Output Format:
Truth_Atoms:
- ...

Claim_Atoms:
- ...

Notes:
- ...
"""

FORWARD_ENTAILMENT_PROMPT = """
You are a LOGICAL ANALYST.

Task:
Assess whether the Internal Fact logically implies the External Claim.

Instructions:
- Assume the Internal Fact is true.
- Determine whether the Claim must also be true.
- Identify assumptions or edge cases where the implication holds or fails.

Output Format:
Assessment:
- Entailed / Partially Entailed / Not Entailed

Reasoning:
- Concise logical explanation.
"""

REVERSE_ENTAILMENT_PROMPT = """
You are a LOGICAL ANALYST.

Task:
Assess whether the External Claim implies the Internal Fact.

Instructions:
- Assume the Claim is true.
- Determine whether the Fact must also be true.
- Identify alternative scenarios where the Claim holds but the Fact does not.

Output Format:
Assessment:
- Entailed / Partially Entailed / Not Entailed

Reasoning:
- Concise explanation focusing on information loss or ambiguity.
"""

OPTIMIST_PROMPT = """
You are the OPTIMIST (Charitable Interpreter).

Goal:
Construct the strongest reasonable case that the External Claim faithfully
represents the Internal Fact.

Guidelines:
- Assume good faith and shared context.
- Treat omissions as simplifications unless critical.
- Highlight compatible interpretations and numerical ranges.

Output:
Argument:
- A persuasive paragraph.
- Include edge cases where this interpretation is valid.
"""

PESSIMIST_PROMPT = """
You are the PESSIMIST (Adversarial Interpreter).

Goal:
Construct the strongest reasonable case that the External Claim is misleading
or mutated relative to the Internal Fact.

Guidelines:
- Assume a skeptical audience.
- Highlight omissions, reframing, or audience effects.
- Focus on how the claim could mislead even if technically compatible.

Output:
Argument:
- A critical paragraph.
- Include edge cases where this interpretation dominates.
"""

SYNTHESIZER_PROMPT = """
You are a SYNTHESIZER.

Inputs include:
- Grounded propositions
- Forward entailment analysis
- Reverse entailment analysis
- Optimist and Pessimist arguments

Task:
- Do NOT choose a winner.
- Identify all reasonable interpretations.
- Explain the assumptions under which each perspective is valid.

Output Format:
Interpretive_Map:
- Perspective 1:
  - Description:
  - Assumptions:
  - Risks:

- Perspective 2:
  - Description:
  - Assumptions:
  - Risks:

Summary:
- What drives disagreement in this case?
"""





def run_agent(model, system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"



# main loop

def test_row(row_index):
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

# read file
    df = pd.read_csv(INPUT_FILE)

    if row_index < 0 or row_index >= len(df):
        print(f"Error: Row {row_index} does not exist. Max row: {len(df)-1}")
        return

    row = df.iloc[row_index]
    internal_fact = row.get("truth", "N/A")
    external_claim = row.get("claim", "N/A")

    case_input = f"""
INTERNAL FACT:
{internal_fact}

EXTERNAL CLAIM:
{external_claim}
"""

    print("\n" + "=" * 70)
    print(f"TESTING ROW {row_index}")
    print("=" * 70)

    # 1. Grounding
    grounding = run_agent(MODEL, GROUNDING_PROMPT, case_input)
    print("\n[GROUNDING]\n", grounding)

    # 2. Forward entailment
    forward = run_agent(MODEL, FORWARD_ENTAILMENT_PROMPT, case_input)
    print("\n[FORWARD ENTAILMENT]\n", forward)

    # 3. Reverse entailment
    reverse = run_agent(MODEL, REVERSE_ENTAILMENT_PROMPT, case_input)
    print("\n[REVERSE ENTAILMENT]\n", reverse)

    # 4. Optimist
    optimist = run_agent(MODEL, OPTIMIST_PROMPT, case_input)
    print("\n[OPTIMIST]\n", optimist)

    # 5. Pessimist
    pessimist = run_agent(MODEL, PESSIMIST_PROMPT, case_input)
    print("\n[PESSIMIST]\n", pessimist)

    # 6. Synthesis
    synthesis_input = f"""
CASE:
{case_input}

GROUNDING:
{grounding}

FORWARD ENTAILMENT:
{forward}

REVERSE ENTAILMENT:
{reverse}

OPTIMIST:
{optimist}

PESSIMIST:
{pessimist}
"""
    synthesis = run_agent(MODEL, SYNTHESIZER_PROMPT, synthesis_input)
    print("\n[SYNTHESIS]\n", synthesis)

    print("\n" + "=" * 70 + "\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-agent faithfulness analysis on a CSV row."
    )
    parser.add_argument(
        "row",
        type=int,
        nargs="?",
        default=0,
        help="Row number to test (default: 0)",
    )

    args = parser.parse_args()
    test_row(args.row)  # test case: which claim to read