import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
# Pricing per 1M tokens: (input, output)
PRICING = {
    "gpt-4.1":      (2.00,  8.00),
    "gpt-4.1-mini": (0.40,  1.60),
    "gpt-5.2":      (1.75, 14.00),
    "o3-mini":      (1.10,  4.40),
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
prompt = "Hello"

for model, (in_price, out_price) in PRICING.items():
    print(f"\n{'=' * 50}")
    print(f"Model: {model}")
    print(f"{'=' * 50}")

    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - start

        msg = response.choices[0].message.content
        usage = response.usage

        input_cost = usage.prompt_tokens * in_price / 1_000_000
        output_cost = usage.completion_tokens * out_price / 1_000_000
        total_cost = input_cost + output_cost

        print(f"Response: {msg}")
        print(f"Time:     {elapsed:.2f}s")
        print(f"Tokens:   {usage.prompt_tokens} in / {usage.completion_tokens} out / {usage.total_tokens} total")
        print(f"Pricing:  ${in_price:.2f} in / ${out_price:.2f} out per 1M tokens")
        print(f"Cost:     ${input_cost:.6f} in + ${output_cost:.6f} out = ${total_cost:.6f} total")
    except Exception as e:
        print(f"Error: {e}")