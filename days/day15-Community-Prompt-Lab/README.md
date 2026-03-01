# Day 15 — Community Prompt Lab

A practical prototype to help the community **experiment publicly**, **remix ideas**, and **share what works**.

## Features
- Prompt challenge submission form.
- Community gallery with type and theme metadata.
- Theme filter for browsing challenge sets.
- 3 voting dimensions: creativity, usefulness, reproducibility.
- One-click **Remix** to fork someone else's prompt.
- Lightweight leaderboard based on total votes.
- Local persistence via `localStorage`.

## Run locally
From this folder:

```bash
python3 -m http.server 8015
```

Then open:

`http://localhost:8015`

## Test locally

```bash
node --test core.test.mjs
```

## Why this fits the goal
It creates a visible loop for experimentation: publish → remix → vote → improve.
