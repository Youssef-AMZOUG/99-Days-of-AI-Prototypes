export const ENTRY_TYPES = ["text", "image", "code", "audio"];

export function sanitizeText(value) {
  return String(value ?? "").trim();
}

export function createEntry(fields, randomId = () => `entry-${Date.now()}`) {
  const title = sanitizeText(fields.title);
  const prompt = sanitizeText(fields.prompt);
  const model = sanitizeText(fields.model);
  const type = ENTRY_TYPES.includes(fields.type) ? fields.type : "text";
  const result = sanitizeText(fields.result);
  const theme = sanitizeText(fields.theme) || "open";

  if (!title || !prompt || !model || !result) {
    return { ok: false, error: "missing_required_fields" };
  }

  return {
    ok: true,
    entry: {
      id: randomId(),
      title,
      prompt,
      model,
      type,
      result,
      theme,
      votes: { creative: 0, useful: 0, repro: 0 },
      createdAt: Date.now(),
    },
  };
}

export function addVote(entries, id, bucket) {
  return entries.map((item) => {
    if (item.id !== id) return item;
    return {
      ...item,
      votes: {
        ...item.votes,
        [bucket]: (item.votes?.[bucket] || 0) + 1,
      },
    };
  });
}

export function totalVotes(votes) {
  return (votes?.creative || 0) + (votes?.useful || 0) + (votes?.repro || 0);
}

export function leaderboard(entries, limit = 5) {
  return [...entries]
    .sort((a, b) => totalVotes(b.votes) - totalVotes(a.votes))
    .slice(0, limit)
    .map((entry) => ({
      id: entry.id,
      title: entry.title,
      points: totalVotes(entry.votes),
    }));
}

export function remixDraft(entry) {
  return {
    title: `${entry.title} (Remix)`,
    prompt: `${entry.prompt}\n\nConstraint: Add one surprising twist.`,
    model: entry.model,
    type: entry.type,
    theme: entry.theme || "open",
    result: "",
  };
}
