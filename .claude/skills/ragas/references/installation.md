# Installation

## Minimal install

```bash
pip install ragas
```

The `ragas` package on PyPI is the canonical name. Verify the version after install:

```bash
python -c "import ragas; print(ragas.__version__)"
```

## Peer dependencies (recommended)

RAGAS uses LangChain wrappers under the hood for most LLM providers. If you'll use the `LangchainLLMWrapper` pattern (which is what the official examples use), pin compatible versions explicitly to avoid mismatched-API errors at evaluate time:

```bash
pip install -U "langchain-core>=0.2,<0.3" "langchain-openai>=0.1,<0.2" openai
```

For a non-LangChain path (RAGAS native factories), see `judge-config.md` — you can use `llm_factory` and skip the LangChain peer deps entirely.

## Pinning RAGAS for production / CI

RAGAS is in active development with breaking changes between minor versions (metric renames, namespace moves like `ragas.metrics` → `ragas.metrics.collections`). For any project where reproducibility matters (CI, eval suite that gates a deploy), pin a specific minor:

```
ragas>=0.2,<0.3
```

When upgrading, re-read `metrics.md` here and run a smoke eval on a known-good dataset to catch silent score shifts.

## Verifying the install works

Smallest possible smoke test — does it import, can it call the judge, does `evaluate()` return a result?

```python
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

dataset = EvaluationDataset.from_list([{
    "user_input": "What is 2+2?",
    "response": "4",
    "retrieved_contexts": ["Basic arithmetic: 2+2 equals 4."],
}])

result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness()],
    llm=LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0)),
)
print(result)  # {'faithfulness': 1.0} (or close to it)
```

If this runs end-to-end and prints a score, your install + credentials are working. If not, check `OPENAI_API_KEY` is set and that the LangChain peer deps are installed.
