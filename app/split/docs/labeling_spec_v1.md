# Labeling Spec v1

## Objective

Convert a user query into a structured normalized representation for retrieval.

## Labels

### `type`

Allowed values:

- `词项`
- `动作短语`

### `core_text`

The query after stripping question-shell noise and obvious scaffolding.

Examples:

- `开心怎么说` -> `开心`
- `寻找帮助怎么讲` -> `寻找帮助`
- `上海话里我喜欢你怎么说` -> `我喜欢你`

### `predicate`

Only used for `动作短语`.

Examples:

- `寻找帮助` -> `寻找`
- `我喜欢你` -> `喜欢`

### `object`

Only used when the object is retrieval-useful.

Delete:

- personal pronouns
- particles
- function words

Keep:

- meaningful nouns with retrieval value

Examples:

- `我喜欢你` -> ``
- `寻找帮助` -> `帮助`
- `学习编程` -> `编程`

### `keywords`

Rules:

- all Chinese only
- deduplicated
- no explanation text
- `词项`: 1 to 5 retrieval-useful synonyms or near-synonyms
- `动作短语`: core predicate, optionally plus useful object

## Special lexical exceptions

Treat as `词项` even if they look decomposable:

- 没用
- 有意思
- 难受
- 开心
- 伤心
- 高兴

## Forbidden output content

- `我 你 他 她 它 我们 你们 他们`
- `给 把 被 对 向 跟 在 从 和`
- `啊 呢 吗 吧 呀`
- `的 地 得`
- non-Chinese filler
- explanations

## Edge-case examples

### Example 1

Input:

`开心怎么说`

Output:

```json
{
  "core_text": "开心",
  "type": "词项",
  "predicate": "",
  "object": "",
  "keywords": ["快乐", "高兴", "愉悦"]
}
```

### Example 2

Input:

`我喜欢你`

Output:

```json
{
  "core_text": "我喜欢你",
  "type": "动作短语",
  "predicate": "喜欢",
  "object": "",
  "keywords": ["喜欢"]
}
```

### Example 3

Input:

`寻找帮助怎么讲`

Output:

```json
{
  "core_text": "寻找帮助",
  "type": "动作短语",
  "predicate": "寻找",
  "object": "帮助",
  "keywords": ["寻找", "帮助"]
}
```
