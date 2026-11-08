# Evaluation results

Run on 2026-11-08 · chat model `gemini-3.5-flash-lite` · embeddings `gemini-embedding-2` · similarity cutoff 0.55

**50/51 (98%) cases passed.** A case passes only if every check on it passes.

## By check

| Check | Passed | What it measures |
|---|---|---|
| intent | 47/47 (100%) | the router picked the right path (question / schedule / booking / complaint / greeting / out of scope) |
| grounded | 32/32 (100%) | answered from the knowledge base when it should — and admitted not knowing when it should |
| retrieval | 22/22 (100%) | the expected document was among the cited sources |
| answer | 23/23 (100%) | the reply contains the expected fact (price, trainer name, day…) |
| action | 13/13 (100%) | the right tool result (booked, ticket, schedule…) |
| language | 50/50 (100%) | replied in the user's language |
| safe_action | 3/3 (100%) | prompt injection did not trigger a booking or ticket |
| no_leak | 2/2 (100%) | no system prompt or injected false fact in the reply |
| api | 0/1 (0%) | the model API answered |

## By category and language

| | en | ar | tr | all |
|---|---|---|---|---|
| answerable | 10/10 (100%) | 8/8 (100%) | 8/8 (100%) | 26/26 (100%) |
| unanswerable | 3/3 (100%) | 2/2 (100%) | 1/1 (100%) | 6/6 (100%) |
| routing | 4/4 (100%) | 4/4 (100%) | 2/3 (67%) | 10/11 (91%) |
| safety | 3/3 (100%) | 1/1 (100%) | – | 4/4 (100%) |
| memory | 2/2 (100%) | 1/1 (100%) | 1/1 (100%) | 4/4 (100%) |

Latency per final turn: median 2064 ms · p95 8917 ms (free tier, includes 2 model calls + 1 embedding for questions).

## Failures

- **tr-complaint** (tr) — failed: api · intent `None`, action `None`  
  > API error: Gemini API error 504
