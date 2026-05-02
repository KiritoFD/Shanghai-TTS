# Query Sources

For this project, the best training mix is:

1. Existing query-like datasets
2. Real business logs if available
3. Teacher-expanded synthetic hard cases

## Recommended sources

### Directly usable or easily adaptable

- `DMetaSoul/chinese-semantic-textual-similarity`
  Useful because parts of the collection include short, colloquial search or dialogue-style texts such as `BUSTM`.
- Chinese QA or short-query datasets on Hugging Face
  Good for harvesting short user-like expressions, then re-filtering to fit this task.

### Useful references

- `JDsearch`
  Real Chinese product search queries from JD. Strong source if you can access the released data or a mirror.
- `Tiangong-ST`
  Chinese web search session data, useful when you need real search-query distributions.
- `QBSUM`
  Real-world Chinese query-based summarization dataset with realistic queries.

## Recommended build strategy

- `60%` existing datasets
- `20%` hard-case templates
- `20%` teacher expansions and paraphrases

This is better than pure synthetic generation because it preserves the distribution of real request text while still covering edge cases like:

- `没用`
- `有意思`
- `我喜欢你`
- `寻找帮助怎么讲`
