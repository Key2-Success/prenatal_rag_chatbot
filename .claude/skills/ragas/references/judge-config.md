# Configuring the judge LLM (and embeddings)

The judge LLM is the model RAGAS uses to *evaluate* answers. Its choice affects scoring quality more than any other RAGAS knob.

## Two rules for judge selection

1. **Cross-vendor by default.** Pick a judge from a *different vendor* than the answer LLM. A model judging output from its own family tends to score generously (self-favouring bias) AND shares failure modes — it can't reliably catch errors it would have made itself. See `pitfalls.md` #2.
2. **At least as strong, ideally stronger** than the answer LLM. A weaker judge can't catch subtle hallucinations a stronger answer model produces. With `temperature=0`, stronger models are also more consistent across runs.

**Always pin to a dated snapshot, not a floating alias.** Floating aliases (e.g. `claude-sonnet-4-5`, `gpt-4o`) silently shift versions and move scores by 0.05+. Use the dated form (`claude-sonnet-4-5-20250929`, `gpt-4o-2024-08-06`) so longitudinal comparisons stay meaningful.

Practical pairings:

| Answer LLM (vendor)       | Recommended judge (vendor)                              |
| ------------------------- | ------------------------------------------------------- |
| `gpt-4.1-nano` (OpenAI)   | `claude-sonnet-4-5-20250929` (Anthropic)                |
| `gpt-4o-mini` (OpenAI)    | `claude-sonnet-4-5-20250929` (Anthropic)                |
| `gpt-4o` (OpenAI)         | `claude-opus-...` (Anthropic) or `claude-sonnet-4-5-...`|
| `claude-sonnet-...` (Anthropic) | `gpt-4o-2024-08-06` or stronger (OpenAI)          |
| `claude-haiku-...` (Anthropic)  | `gpt-4o-mini` or stronger (OpenAI)                |
| open-source 7-13B         | `gpt-4o-mini` (OpenAI) or `claude-haiku-...` (Anthropic) — minimum |

If you can't use a cross-vendor judge (e.g. only one provider's API is available), document the reason in code and pick a meaningfully stronger model from the same vendor (e.g. `gpt-4o` judging `gpt-4o-mini`). Never judge with the same or weaker tier than the answer LLM.

**Always set `temperature=0` on the judge.** Eval scores must be reproducible across runs.

## Two configuration patterns

### Pattern A: LangChain wrapper (most common)

Used in the official RAGAS getting-started examples. Wrap any LangChain-compatible chat model:

```python
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate

judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), ResponseRelevancy()],
    llm=judge,
)
```

For non-OpenAI providers, swap the LangChain class:

```python
from langchain_anthropic import ChatAnthropic           # Anthropic
from langchain_aws import ChatBedrockConverse            # AWS Bedrock
from langchain_google_genai import ChatGoogleGenerativeAI  # Google
```

### Pattern B: RAGAS native factory

Skips the LangChain dep entirely. Newer RAGAS-native path:

```python
from ragas.llms import llm_factory
from openai import OpenAI

judge = llm_factory(
    "gpt-4o-mini",
    client=OpenAI(),
)
```

Use Pattern B if you want to avoid langchain peer deps. Use Pattern A if you're already on a LangChain-based stack or want maximum example-coverage in the docs.

## Embeddings

Some metrics (e.g. `ResponseRelevancy`, `SemanticSimilarity`) need embeddings in addition to the judge LLM. Pass them via the `embeddings=` arg to `evaluate()`:

```python
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

embedder = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

result = evaluate(
    dataset=dataset,
    metrics=[ResponseRelevancy()],
    llm=judge,
    embeddings=embedder,
)
```

If you don't pass `embeddings`, RAGAS uses a default — usually OpenAI's. Pass explicitly when (a) using a non-OpenAI judge, (b) you want to pin a specific embedding model for reproducibility, or (c) running offline.

## Concurrency / batch size

`evaluate()` accepts a `RunConfig` to control parallel calls. Default concurrency is conservative; bump it on a fast machine + uncapped API limits, lower it if you're hitting rate-limit errors:

```python
from ragas.run_config import RunConfig

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness()],
    llm=judge,
    run_config=RunConfig(max_workers=8, timeout=180),
)
```

Don't crank `max_workers` blindly — RAGAS's default exists because each metric makes 3-10 sequential calls per sample, and aggressive parallelism can blow through rate limits or empty wallets fast. Start at 4-8.
