
# FactTrace Hackathon @ University of Cambridge

### 🧠 The Agentic Consensus Challenge

**Welcome.**
Your mission is to build a jury of AI agents that debates the truth.
We don't want a black box that just says "True" or "False." We want to see agents **disagree, argue, and negotiate** to reach a verdict.

---

## 🚩 The Challenge: "Faithful or Mutated?"

You will be given pairs of statements:

1. **Internal Fact:** A source truth (e.g., a specific statistic from a report).
2. **External Claim:** A statement derived from that fact (e.g., a tweet, headline, or summary).

**Your Question:**
Is the external claim a faithful representation of the internal fact — or is it a mutation?

**The Catch:**
The data is ambiguous by design. "Technically true" isn't enough. Your agents need to figure out if the *meaning* has shifted (e.g., through exaggeration, missing context, or causal confusion).

---

## ⚖️ What to Build

A single AI can answer this. **A jury explains it.**
Your primary goal is to design the **interaction** between agents.

* **Create Roles:** Don't just clone the same agent. Build a "Sceptic," a "Pedantic Fact-Checker," or a "Common Sense Judge."
* **Let Them Fight:** What happens when they disagree? Do they vote? Do they compromise?
* **The Verdict:** Your system must output a final decision and a transparent reason why.

---

## 📁 The Data

* **Your Team, Your Data.** You will find a dataset file that matches your team name.
* **Pick your battles.** You do not need to process the whole file. **Decide on a subset size** that works for you (we recommend **5 pairs**).
* **Focus.** Choose the cases that are interesting and likely to spark a good argument between your agents.

---

## 🔑 API Keys & Credits

We are making this easy for you:

1. **You get a key.** We will hand each team a pre-loaded API key at the start.
2. **It has a limit.** The credit is generous but finite.
3. **Don't burn it.**
* **Development:** Use cheap models (e.g., `gpt-4o-mini`) while testing your loops.
* **The Demo:** Switch to the flagship models (e.g., `gpt-4o`) only for your final presentation.



---

## 🏁 The 5-Minute Pitch

At the end, you have **5 minutes** to show us what you built.
**Keep it simple:**

1. **Show the Debate:** We want to see the agents arguing in your terminal or notebook.
2. **Show the Verdict:** Did they agree? Why?
3. **The "Why":** Explain why your multi-agent approach is better than a single prompt.

---

### 🏆 Judging Criteria

**1. Agent Design (30%) 🧠**
Do your agents have distinct roles that actually debate? We look for meaningful interaction—not just a chain of prompts—and evidence that the team beats a single AI.

**2. Reasoning & Explanation (30%) 🧾**
Does the system explain *why* it reached a verdict? We value honest handling of uncertainty and clear insights into why a case is ambiguous.

**3. Data Understanding (20%) 🔍**
Did you choose a strategic subset? We look for thoughtful selection that highlights interesting mutations rather than just processing random rows.

**4. Demo Clarity (20%) 🎤**
Can you explain your solution in 5 minutes? A great demo is structured, easy to follow, and teaches the judges something new.

**Good luck. Start building.**
