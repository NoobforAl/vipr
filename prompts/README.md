# Prompt categories

One JSON file per category: `prompts/<id>.json`. Users can edit these or add new files (the file name without
`.json` is the id used in the app and in the llm-worker input `about.category`).

| field | meaning |
|---|---|
| `name` | label shown in the UI |
| `description` | one-line help text |
| `fix_prompt` | extra paragraph added to the text-cleanup system prompt (domain, tone, what to keep) |
| `translate_prompt` | extra paragraph for the translation step (empty until translation exists) |
| `suggested_glossary` | terms the UI offers to add to the glossary (NOT merged automatically) |

The fixed base prompt (rules for fixing words, punctuation, fillers, speaker labels) is in code
(`engine::prompts`), and the safety guards (similarity, line count, speaker-flip caps ...) are code too:
a category can steer the editor but can never switch the guards off.
