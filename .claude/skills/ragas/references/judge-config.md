# Configuring the judge LLM (and embeddings)

The judge LLM is the model RAGAS uses to *evaluate* answers. Its choice affects scoring quality more than any other RAGAS knob.

## Why a stronger judge than the answer LLM

The single most important rule: **the judge must be at least as strong as, ideally stronger than, the model that produced the answer.** Reasons:

1. **Self-favouring bias.** A model judging its own output tends to score it generously. Using a different model breaks the loop.
2. **Detection ceiling.** A weaker judge cannot reliably catch errors a stronger model would. If your answer LLM produces subtle hallucinations, a peer-strength judge will miss them.
3. **Determinism.** Stronger models (with `temperature=0`) are more consistent across runs. Eval scores need to be reproducible to be useful.

Practical pairings:

| Answer LLM        | Recommended judge |
| ----------------- | ----------------- |
| `gpt-4o`          | `gpt-4o` or stronger (Claude Sonnet, etc.) |
| `gpt-4o-mini`     | `gpt-4o`          |
| `gpt-4.1-nano`    | `gpt-4o-mini` or `gpt-4.1-mini` |
| open-source 7-13B | `gpt-4o-mini` minimum |

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
