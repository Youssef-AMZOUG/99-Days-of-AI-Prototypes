import test from 'node:test';
import assert from 'node:assert/strict';

import { addVote, createEntry, leaderboard, remixDraft } from './core.js';

test('createEntry returns error for missing required fields', () => {
  const result = createEntry({ title: 'x' }, () => 'id-1');
  assert.equal(result.ok, false);
  assert.equal(result.error, 'missing_required_fields');
});

test('createEntry builds valid entry payload', () => {
  const result = createEntry(
    {
      title: '  Prompt challenge  ',
      prompt: '  Prompt body ',
      model: ' GPT-4 ',
      type: 'text',
      theme: 'education',
      result: ' output ',
    },
    () => 'id-2',
  );

  assert.equal(result.ok, true);
  assert.equal(result.entry.id, 'id-2');
  assert.equal(result.entry.title, 'Prompt challenge');
  assert.equal(result.entry.theme, 'education');
  assert.deepEqual(result.entry.votes, { creative: 0, useful: 0, repro: 0 });
});

test('addVote increments expected vote bucket', () => {
  const entries = [{ id: 'a', votes: { creative: 0, useful: 1, repro: 2 } }];
  const next = addVote(entries, 'a', 'useful');
  assert.equal(next[0].votes.useful, 2);
  assert.equal(next[0].votes.repro, 2);
});

test('leaderboard sorts by total points descending', () => {
  const board = leaderboard([
    { id: 'a', title: 'A', votes: { creative: 2, useful: 0, repro: 0 } },
    { id: 'b', title: 'B', votes: { creative: 0, useful: 3, repro: 1 } },
    { id: 'c', title: 'C', votes: { creative: 1, useful: 1, repro: 1 } },
  ]);

  assert.equal(board[0].id, 'b');
  assert.equal(board[0].points, 4);
});

test('remixDraft appends remix guidance', () => {
  const draft = remixDraft({
    title: 'Original',
    prompt: 'Prompt',
    model: 'Model',
    type: 'text',
    theme: 'open',
  });

  assert.match(draft.title, /Remix/);
  assert.match(draft.prompt, /surprising twist/);
  assert.equal(draft.result, '');
});
