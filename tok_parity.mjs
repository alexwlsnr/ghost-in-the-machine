// Parity: browser makeTokenizer() must reproduce the Python tokenization for
// both BPE schemes (char regression + new bytelevel). Compares JS encode()
// against /tmp/tok_refs.json produced by the Python references.
import { readFileSync } from 'fs';
import { makeTokenizer } from './dist/bpe_tokenizer.js';

const refs = JSON.parse(readFileSync('/tmp/tok_refs.json', 'utf8'));
const tests = refs.tests;

// char scheme: from a deployed char manifest's tokenizer section
const charTok = makeTokenizer(JSON.parse(readFileSync('/tmp/char_tok.json', 'utf8')));
// bytelevel scheme: from the freshly serialized byte-level manifest
const blMan = JSON.parse(readFileSync('./dist/model_wisp_bytelevel_ep21.json', 'utf8'));
const blTok = makeTokenizer(blMan.tokenizer);

let allOk = true;
for (const [name, tok] of [['char', charTok], ['bytelevel', blTok]]) {
  console.log(`\n━━━ ${name} (type=${name === 'char' ? 'char/legacy' : blMan.tokenizer.type})`);
  tests.forEach((t, i) => {
    const got = tok.encode(t);
    const want = refs[name][i];
    const ok = got.length === want.length && got.every((v, j) => v === want[j]);
    allOk &&= ok;
    console.log(`  ${ok ? 'OK ' : 'MISMATCH'} | ${t}`);
    if (!ok) { console.log('     js  :', got); console.log('     py  :', want); }
  });
}
console.log('\n' + (allOk ? '✅ ALL MATCH — JS tokenizers reproduce Python exactly' : '❌ MISMATCHES'));
process.exit(allOk ? 0 : 1);
