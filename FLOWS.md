# How ScholarGraph turns a question into an answer

This document walks through what actually happens between the moment a user types a query in our UI and the moment a synthesized markdown answer appears back on the page. It's written for the team — no code-level detail, but enough to picture every transformation the data goes through.

---

## What we're searching over

Before any user query exists, Part A has already done its work and populated our Postgres database. The unit of search is a **chunk**:

- A chunk is one passage of one paper — typically a section or a portion of one.
- We have ~97,000 of them, drawn from CS papers fetched from OpenAlex.
- Each chunk carries the **paper's metadata** (title, year, authors, DOI, PDF URL, domain/field/subfield) and the **chunk's own metadata** (section title, section type like `method`/`limitation`/`discussion`, position within the paper).
- Each chunk also carries a **384-dimensional embedding vector** of its text, computed once on ingest using `sentence-transformers/all-MiniLM-L6-v2`. This is what makes semantic search possible.

Everything sits in a single denormalized `chunks` table, so any search returns rows that already have all the context we need — no joins.

---

## The two entry points

The frontend gives the user two ways to ask for something:

1. **Three analyze buttons** — "Find Research Gaps", "Generate Table of Contents", "Synthesize Methodologies". Each takes a topic/question and produces a structured, multi-paper analysis.
2. **A chat panel** — free-form conversation. The system decides for itself whether the message wants one of the three analyses, or just a plain-text reply.

Under the hood, the buttons and the chat share the same machinery. The chat just adds a routing step in front.

---

## Flow 1 — Pressing an analyze button

Let's say the user types "graph neural networks for molecular property prediction" and clicks **Find Research Gaps**.

### Step 1. The request leaves the frontend

The frontend sends a JSON POST to `/analyze/gaps`:

```
{ "query": "graph neural networks for molecular property prediction",
  "top_k_per_query": 8 }
```

(The same shape goes to `/analyze/toc` and `/analyze/methodologies` for the other two buttons.)

### Step 2. The backend expands the query

We rarely want to search with the user's exact wording — it might be too narrow, too colloquial, or miss the precise technical terms papers actually use. So we first ask an LLM to **rewrite the user's question into 3 alternative search queries** using academic terminology.

The expansion prompt tells the LLM to:

- Identify the core technical concepts in the user's input.
- Produce 3 reformulated queries that focus on methodology, results, and constraints.
- Use professional academic vocabulary.

So the example above might become something like:
- "graph convolutional networks applied to molecular property regression tasks"
- "message-passing neural networks for chemical property prediction benchmarks"
- "limitations of GNN architectures in cheminformatics QSAR modeling"

**Goal:** cast a wider, more precisely-worded net. We trade one fuzzy human query for three sharp technical ones.

If the LLM's output isn't parseable, we fall back to using the user's original query as-is.

### Step 3. The backend runs 3 hybrid searches in parallel

For each of the 3 expanded queries, we run a **hybrid semantic + keyword search** against the database, asking for the top 8 chunks per query.

**What goes into each search:** one of the 3 expanded query strings — just that string, as plain text. We do *not* search with the user's original query, nor with all 3 expansions concatenated. Each expansion runs its own independent search and we merge the results afterwards. So with input "graph neural networks for molecular property prediction" we end up with 3 separate retrievals, each driven by one rewritten phrasing of that topic.

For the gaps button specifically, we also filter to only chunks whose `section_type` is `limitation`, `discussion`, or `conclusion` — because those are the sections where authors actually talk about what's missing or unresolved. The other buttons use different filters:

- **Gaps** → `limitation`, `discussion`, `conclusion`
- **ToC** → no filter (the outline needs all sections)
- **Methodologies** → `method`, `result`

### Step 4. Dedupe and (maybe) fall back

The 3 searches return up to 3 × 8 = 24 chunks. Many will overlap, so we dedupe by `chunk_id`.

We then count how many *distinct papers* survived. If it's fewer than 3, the filter was probably too aggressive (e.g. very few papers in our DB have `limitation` chunks on this topic). In that case we run the same 3 searches again **without** the section_type filter, merge in the results, and dedupe again. This guarantees we always have enough material to compare across papers, even if some of it isn't in the ideal section type.

### Step 5. Group chunks by paper

The chunks that come out of search are sorted by retrieval score. But the synthesis step wants to reason **per paper** — "what does paper X say about gap Y" — so we re-organize:

- Bucket chunks by `paper_id`.
- Attach the paper's metadata once (title, year, authors, etc.) instead of repeating it on every chunk.
- Sort papers by their best chunk's score, and sort chunks within each paper by score too.
- Stamp a per-chunk `pdf` URL onto every chunk, pointing at `/chunks/{chunk_id}/pdf` — the frontend's PDF viewer uses this directly.

We end up with a list of papers, each with a small bundle of its most relevant chunks.

### Step 6. Synthesize

Now we hand the grouped papers to the LLM with a synthesis prompt tailored to the button. Each synthesis prompt is built the same way:

1. We format the papers into a `[Context Data]` block — one entry per paper with its metadata header followed by its chunks (each labelled with section title, section type, and position).
2. We append a `[Strict Grounding Rules]` block that's identical across all three syntheses:
   - **No external knowledge** — only use what's in the provided chunks.
   - **Missing-info fallback** — if the data can't support an answer, the LLM must say so explicitly rather than guess.
   - **Citation enforcement** — every claim must be followed by `[Title (Year), §section_title]`.
   - **Verbatim fidelity** — when quoting technical content, stay close to the chunk's original wording.
3. We append the **task** — different per button:
   - **Gaps** asks for per-paper limitations, contradictions across papers, synergies between future-work suggestions, and one "research silence" no paper addresses.
   - **ToC** asks for a hierarchical outline (cross-cutting themes, chapters, subsections) that a literature-review chapter could follow.
   - **Methodologies** asks for a per-paper methodology profile and a cross-paper comparison matrix.

The LLM returns a single markdown string. That's the analysis.

### Step 7. Send the response back

The backend replies with:

```
{ "query": "<the original user query>",
  "expanded_queries": [...3 strings...],
  "papers": [
    { paper metadata...,
      "chunks": [ { chunk_id, pdf, section_title, section_type, position, text, score }, ... ] },
    ...
  ],
  "analysis": "<the markdown synthesis>" }
```

The frontend renders the `analysis` markdown, lists the source papers, and lets the user click any chunk to open the original PDF inline (via the `pdf` field on each chunk).

---

## Flow 2 — Typing in chat

The chat panel sends the whole conversation history (a list of `{role, content}` messages) to `/chat`. The job is to figure out what the user actually wants, and then dispatch to one of five destinations: a grounded Q&A answer (the default), one of the three specialized analyses, or a plain LLM-only reply.

### Step 1. The router decides

We ask an LLM a single question: given the whole conversation, classify it. The router prompt tells the LLM to return a JSON object with two fields:

```
{ "pipeline": "gaps" | "toc" | "methodologies" | "qa" | "text",
  "search_query": "<a rewritten standalone query>" }
```

- **`pipeline`** picks the destination.
  - `gaps`, `toc`, `methodologies` — fire the corresponding specialized pipeline (same as pressing the button). Only chosen when the user EXPLICITLY asks for a limitations critique, a discussion outline, or a methodology comparison.
  - **`qa` is the default** for any other research question — factual lookups about specific papers, "what does X say about Y", broad conceptual questions, comparisons that aren't a methodology matrix. Same retrieval machinery as the three buttons, but **unfiltered** (answers can live in any section) and with a **neutral query expansion** (the buttons' expansion biases toward methodology/results/constraints, which is wrong for general Q&A).
  - `text` means "no retrieval, just answer conversationally" — reserved for unmistakable chitchat (greetings, gratitude, meta-questions about the assistant). Broad research questions go to `qa`, not `text`.
- **`search_query`** is a self-contained reformulation of what the user is really asking *right now*. It exists to handle multi-turn chat where the latest message alone (e.g. "what about the methods?") doesn't make sense without context.

The prompt has explicit rules around how to do the rewrite:

- **Topic-shift detection** — if the latest message moves to a new technical topic (e.g. from "Transformers" to "NFTs"), discard everything before and start fresh.
- **Pronoun-only inheritance** — only carry topic from earlier turns when the user actually used a pronoun ("it", "they") or a follow-up phrase ("how about…", "and the…"). Otherwise treat each turn as standalone.
- **No hallucinated terms** — the rewrite may only contain technical terms the user actually said or clearly implied.
- **Safety fallback** — pick `text` ONLY for unmistakable chitchat (greetings, gratitude, meta-questions). Broad conceptual research questions go to `qa` so they get grounded.
- **Bias toward grounding** — when in doubt between `qa` and a specialized synthesis, pick `qa`. When in doubt between `qa` and `text`, pick `qa`. The conservative answer is always "let's check the database."

If parsing the LLM's response fails for any reason, we default to `("text", <the user's last message verbatim>)`.

### Step 2a. If the router picked `text`

We call the LLM once more with just the conversation history and return its answer. No database hit, no retrieval, no expansion. Used for "hi", "thanks", "what can you do?", etc.

### Step 2b. If the router picked `gaps` / `toc` / `methodologies`

We run **exactly the same pipeline as the corresponding button**, using the rewritten `search_query` as the input instead of the user's last message. From here on, the flow is identical to Flow 1, Steps 2–6.

### Step 2c. If the router picked `qa`

We run a 4th pipeline that mostly mirrors Flow 1 but with two specific differences:

- **Neutral query expansion.** Instead of the buttons' expansion prompt (which says "focus on methodology, results, and constraints"), the qa pipeline uses a more neutral variant that asks the LLM to paraphrase the question across 3 academic angles (entity-focused, relation/property-focused, broader-context) without biasing toward any specific section type.
- **Unfiltered retrieval.** No `section_type` filter — the answer to "what datasets are commonly used for evaluating GNNs?" could live in a `method`, `result`, `introduction`, or even `other` chunk, so we let any chunk compete.

Otherwise it's the same pipeline: expand → 3 parallel searches → dedupe → group by paper → synthesize. The synthesis prompt is also a new one, because the qa task is different from the three fixed analytical tasks: instead of a fixed framing (e.g. "find limitations across papers"), the qa synthesis prompt takes the user's actual question as input and tells the LLM to **answer it directly, leading with the answer and backing it with citations** from the retrieved chunks. The same grounding rules apply — no external knowledge, missing-info fallback, mandatory `[Title (Year), §section_title]` citations.

**Important — what we semantically search with in chat:** the **full chat history never touches the database**. The conversation is only used in one place: as input to the router's LLM call, which produces the single rewritten `search_query` string. From that point onward, only that one string flows into the retrieval pipeline. It gets expanded into 3 variants, and each of those variants is what gets embedded and matched against chunks. So when the user types a follow-up like "what about the methods?", we don't embed that phrase — we embed the router's rewrite (e.g. "graph neural network methodology for molecular property prediction").

### Step 3. The response

The chat endpoint always returns:

```
{ "pipeline": "gaps" | "toc" | "methodologies" | "qa" | "text",
  ...pipeline-specific fields (analysis, papers, expanded_queries)
     or text-specific fields (the assistant message) }
```

The frontend uses the `pipeline` field to decide how to render: a structured analysis card with citations, or a plain chat bubble.

---

## Zooming in: how the semantic search actually works

When the pipeline runs a search, this is what happens for each query string:

1. **Embed the query.** The query string is passed through the same `all-MiniLM-L6-v2` model that embedded our chunks on ingest. We get a 384-dimensional vector.
2. **Score every (eligible) chunk.** For each chunk in the table, the database computes a score:

   ```
   score = (1 - cosine_distance) × 0.7  +  ts_rank(text, query) × 0.3
   ```

   - The first term is **semantic similarity**: how close the query vector is to the chunk's pre-computed vector, in cosine space. This catches paraphrases and conceptually related text even when no words overlap.
   - The second term is **keyword relevance**: PostgreSQL's full-text search rank against the chunk's text. This rewards exact term matches (good for acronyms, proper nouns, and specific technical vocabulary).
   - The 70/30 split is the default; semantic dominates but keyword breaks ties when the user uses a very specific term.
3. **Apply filters.** If the pipeline asked for specific `section_type`s (or any other filter like year, domain, paper_id), those become a SQL `WHERE` clause and only matching chunks compete.
4. **Take the top K** by score and return them as plain rows.

Two indexes make this fast at ~97k chunks:

- An **HNSW index** on the embedding column — approximate nearest-neighbor lookup without scanning every row.
- A **GIN index** on the chunk text's `tsvector` (kept in sync by a trigger) — fast full-text matching.

**What text are we semantically matching, on both sides?**

- **Database side (the haystack):** the chunk's own text — i.e. the actual passage from the paper. We are *not* matching against titles, abstracts, or summaries; we're matching against representations of the paper passages themselves.
- **Query side (the needle):** a single short string per search call. From a button press it's one of the 3 expanded queries. From a chat message it's also one of the 3 expanded queries — but the input *to* that expansion came from the router's rewrite of the conversation, not from the user's last message verbatim. **The full chat history is never embedded**; it only feeds the router LLM.

That's why hybrid scoring matters: a query like "transformer attention" might semantically match a chunk that says "self-attention mechanism" without sharing words, while still being keyword-boosted by a chunk that uses the exact phrase.

---

## Summary diagram

```
USER (chat)                                       USER (button)
   |                                                  |
   v                                                  |
/chat                                             /analyze/{gaps|toc|methodologies}
   |                                                  |
   v                                                  |
router.decide  --pipeline=text--------->  LLM-only reply
   |
   |--pipeline=qa ----------------------> general expansion + unfiltered retrieval
   |                                                  |
   |--pipeline=gaps|toc|methodologies-------------+   |
   |        + rewritten search_query              |   |
   v                                              v   v
                            _collect_papers
                                   |
                                   v
                       expand_query / general_expand_query
                                   |        (1 query -> 3 queries, LLM —
                                   |         neutral variant for qa)
                                   v
                       hybrid_search × 3       (parallel; section-filtered for
                                   |            gaps/methods, unfiltered for toc/qa)
                                   v
                            dedupe + fallback   (if <min_distinct_papers,
                                   |            retry unfiltered — no-op for
                                   |            toc/qa which are already unfiltered)
                                   v
                       group chunks by paper    (each chunk gets a pdf URL)
                                   |
                                   v
                            synthesize with LLM (gap | toc | methodology | qa prompt
                                                 + grounding rules + per-paper context;
                                                 qa also receives the user's question)
                                   |
                                   v
                            return analysis + papers
                                   |
                                   v
                              FRONTEND renders
```

---

## A few useful invariants

- **Every chunk in a response carries a `pdf` link.** The frontend never needs to construct the proxy URL itself. The link points at our backend, which fetches the publisher's PDF server-side with browser-like headers and streams it back — bypassing CORS and bot-detection that would block a direct browser fetch.
- **Every claim in the synthesis is supposed to carry a citation.** The grounding rules force the LLM to attach `[Title (Year), §section_title]` to every statement, so the frontend can deep-link a claim back to a specific chunk.
- **The buttons and the chat share the exact same pipelines.** If you fix a bug or improve a prompt in `pipeline.py` / `workflows.py` / `prompts.py`, both surfaces benefit at once.
- **No HTTP hop between intelligence and retrieval.** The pipelines call the search function directly in-process, so there's no internal API to maintain.
