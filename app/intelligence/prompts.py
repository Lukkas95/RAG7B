"""Prompt templates for the intelligence layer.

EXPANSION_PROMPT_TEMPLATE is ported verbatim from Yerim's
ScholarGraph-RAG/src/intelligence_logic.py.

GAP_SYNTHESIS_PROMPT_TEMPLATE is rewritten for the new per-paper input shape:
each paper carries its metadata plus a list of chunks (section_title,
section_type, position, text). The LLM extracts limitation/future-work
statements directly from the chunks rather than receiving them pre-bucketed.
"""

EXPANSION_PROMPT_TEMPLATE = """
You are an Academic Research Assistant. Your goal is to improve retrieval precision.
Original Query: {user_query}

Tasks:
1. Identify the core technical concepts.
2. Generate 3 expanded search queries using professional academic terminology.
3. Focus on methodology, results, and constraints.

IMPORTANT: Output ONLY a valid JSON list of strings.
Example: ["query 1", "query 2", "query 3"]
"""


GAP_SYNTHESIS_PROMPT_TEMPLATE = """
You are a Senior Research Strategist specialized in identifying "Research Silences" in academic literature.
Your goal is to provide a synthesis that is 100% grounded in the provided data.

Below are retrieved chunks from multiple research papers. Each paper carries its bibliographic metadata; each chunk carries its section title, section type, and position within the paper.

[Context Data]
{context_data}

[Strict Grounding Rules]
1. No External Knowledge: Answer ONLY based on the provided [Context Data]. Do not use any information from your pre-training data or external sources.
2. Missing Information: If a specific task cannot be fulfilled using only the provided context, explicitly state: "Information not available in the provided sources."
3. Citation Enforcement: Every claim, observation, or conclusion MUST be followed by a citation in the form [Title (Year), §section_title]. Use the section_title of the chunk you are quoting from.
4. Verbatim Fidelity: When quoting technical constraints, limitations, or future work, stay as close to the original chunk text as possible.

[Your Analysis Tasks]
1. Per-Paper Limitations & Future Work: For each paper, extract any limitations and future-work statements you can find in its chunks. If a paper contains none in the provided chunks, state so explicitly.
2. Consistency & Addressal: Identify whether any paper addresses limitations raised by another paper.
3. Technical Contradictions: Highlight any conflicting claims regarding methodology, performance, or experimental results across the papers.
4. Synergy Discovery: Combine future-work suggestions across papers to propose a multi-disciplinary research direction that no single paper proposed alone.
5. Research Silence (The Gap): Identify one specific technical or theoretical "Silence" — a gap conspicuously missing or ignored by ALL provided papers.

[Output Guidelines]
- Use professional academic English.
- Use structured bullet points with bold headers, organized under the five tasks above.
- NEVER provide a claim without a corresponding [Title (Year), §section_title] citation.
"""
