"""Prompts. Two families:
  * the audited paper's prompts, loaded verbatim from its repository (needed for H1 comparability);
  * our own prompts, which ask for P(YES) instead of a 0..1 "confidence" (rule 6) and allow INSUFFICIENT.
Freeze the wording with the preregistration commit before any run.
"""
import importlib.util
import json
import os

SYSTEM = (
    "You resolve prediction market questions after the market has closed. Judge only from the evidence provided "
    "in the message. Do not use your own memory of events. Follow the resolution rules literally. "
    "Answer with a single JSON object and nothing else."
)

_HEADER = (
    "Question: {question}\n"
    "Resolution rules: {criteria}\n"
    "Market close time: {close_time}\n\n"
    "Evidence:\n{evidence}\n\n"
)

PROB = _HEADER + (
    "Task: estimate the probability that this market resolves YES, given only the evidence above.\n"
    'Return JSON: {{"p_yes": <number from 0 to 1>, "reasoning": "<at most 80 words>"}}'
)

ABSTAIN = _HEADER + (
    "Task: decide how the market resolves, given only the evidence above. Answer INSUFFICIENT when the evidence "
    "does not show how the question ended (for example: only previews, forecasts or schedules, or nothing from "
    "the resolution window).\n"
    'Return JSON: {{"decision": "YES" | "NO" | "INSUFFICIENT", "p_yes": <number from 0 to 1>, '
    '"reasoning": "<at most 80 words>"}}'
)

CLOSED_SYSTEM = (
    "You answer questions about past events from your own knowledge. Answer with a single JSON object and nothing else."
)
CLOSED = (
    "Question: {question}\nResolution rules: {criteria}\nMarket close time: {close_time}\n\n"
    "Task: from your own knowledge only, estimate the probability that this market resolved YES. "
    'If you do not know, give the probability you would assign before the event.\n'
    'Return JSON: {{"p_yes": <number from 0 to 1>, "reasoning": "<at most 40 words>"}}'
)

DEBATE_R2 = _HEADER + (
    "Your first-round answer:\n{own}\n\n"
    "Answers of the other agents, who saw the same evidence:\n{peers}\n\n"
    "Task: reconsider. Change your answer only if another agent points to something in the evidence that you missed "
    "or misread.\n"
    'Return JSON: {{"decision": "YES" | "NO", "p_yes": <number from 0 to 1>, "changed": true | false, '
    '"reasoning": "<at most 80 words>"}}'
)


def fmt_peers(peers):
    """peers: list of dicts with decision, p_yes, reasoning. Labels are neutral and order should be shuffled by caller."""
    return "\n\n".join(
        f"AGENT {chr(65 + i)}:\n" + json.dumps({k: p.get(k) for k in ("decision", "p_yes", "reasoning")}, ensure_ascii=False)
        for i, p in enumerate(peers)
    )


def load_paper_prompts(repo_path: str) -> dict:
    """Verbatim prompts of arXiv:2605.30802 from a clone of its repository (Architecture A and debate rounds)."""
    def _load(rel, name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(repo_path, rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    a = _load("src/agents/prompts.py", "paper_prompts_a")
    b = _load("src/resolution/debate/prompts.py", "paper_prompts_b")
    return {
        "A_SYSTEM": a.SYSTEM_PROMPT, "A_USER": a.USER_MESSAGE_TEMPLATE,
        "R1_SYSTEM": b.ROUND_1_SYSTEM_PROMPT, "R1_USER": b.ROUND_1_USER_TEMPLATE,
        "R2_SYSTEM": b.ROUND_2_SYSTEM_PROMPT, "R2_USER": b.ROUND_2_USER_TEMPLATE,
    }
