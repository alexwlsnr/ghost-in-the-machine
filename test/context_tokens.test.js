/**
 * Unit tests for buildContextTokens() — pure function, no model needed.
 *
 * Verifies that conversation history is prepended correctly and oldest turns
 * are dropped when the context budget is exceeded.
 *
 * Run: node --test test/context_tokens.test.js
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildContextTokens, encode } from '../dist/tier2_transformer.js';

const SEP = 1;

test('no history: same as bare encode(query) + SEP', () => {
  const tokens = buildContextTokens([], 'HELLO', 128);
  const expected = [...encode('HELLO'), SEP];
  assert.deepEqual(tokens, expected);
});

test('one turn of history prepended before query', () => {
  const history = [{ q: 'HI', r: 'HEY' }];
  const tokens = buildContextTokens(history, 'HOW ARE YOU', 256);
  // Expected: HI[SEP]HEY[SEP]HOW ARE YOU[SEP]
  const expected = [
    ...encode('HI'), SEP,
    ...encode('HEY'), SEP,
    ...encode('HOW ARE YOU'), SEP,
  ];
  assert.deepEqual(tokens, expected);
});

test('two turns of history in chronological order', () => {
  const history = [
    { q: 'HI', r: 'HEY' },
    { q: 'HOW ARE YOU', r: 'GREAT' },
  ];
  const tokens = buildContextTokens(history, 'TELL ME A JOKE', 256);
  // Expected: HI[SEP]HEY[SEP]HOW ARE YOU[SEP]GREAT[SEP]TELL ME A JOKE[SEP]
  const hiBytes    = encode('HI');
  const heyBytes   = encode('HEY');
  const howBytes   = encode('HOW ARE YOU');
  const greatBytes = encode('GREAT');
  const jokeBytes  = encode('TELL ME A JOKE');
  const expected = [
    ...hiBytes, SEP, ...heyBytes, SEP,
    ...howBytes, SEP, ...greatBytes, SEP,
    ...jokeBytes, SEP,
  ];
  assert.deepEqual(tokens, expected);
});

test('oldest turn dropped when budget exceeded', () => {
  // Build history where both turns together would overflow, but newest fits
  const longTurn = 'A'.repeat(50);
  const history = [
    { q: longTurn, r: longTurn },  // old — 102+ tokens, should be dropped
    { q: 'HI', r: 'HEY' },        // recent — fits
  ];
  const maxLen = 30;  // tight budget
  const tokens = buildContextTokens(history, 'BYE', maxLen);
  assert.ok(tokens.length <= maxLen, `tokens (${tokens.length}) exceeded maxLen (${maxLen})`);
  // The long old turn must not be present
  assert.ok(!tokens.includes('A'.charCodeAt(0)) || tokens.filter(t => t === 'A'.charCodeAt(0)).length < 5,
    'old long turn should have been dropped');
});

test('query always present even when history cannot fit', () => {
  const history = [{ q: 'A'.repeat(200), r: 'B'.repeat(200) }];
  const tokens = buildContextTokens(history, 'HELLO', 64);
  const queryTokens = [...encode('HELLO'), SEP];
  // Last elements should be the query
  const tail = tokens.slice(-queryTokens.length);
  assert.deepEqual(tail, queryTokens);
});

test('tokens end with SEP (response zone marker)', () => {
  const tokens = buildContextTokens([], 'HELLO', 128);
  assert.equal(tokens[tokens.length - 1], SEP);
});

test('history turns uppercased automatically', () => {
  const history = [{ q: 'hello', r: 'hey there' }];
  const tokens = buildContextTokens(history, 'BYE', 256);
  const withLower = buildContextTokens([{ q: 'HELLO', r: 'HEY THERE' }], 'BYE', 256);
  assert.deepEqual(tokens, withLower);
});
