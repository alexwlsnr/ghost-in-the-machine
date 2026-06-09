// BPE tokenizer loaded from the model manifest's `tokenizer` section.
// Encode is O(text_len × n_merges) — fast enough for short prompts.
// Decode is O(1) per token via id→string map.

export interface BPEData {
  vocab: Record<string, number>;
  id_to_token: Record<string, string>;
  merges: [string, string][];
}

export class BPETokenizer {
  readonly PAD = 0;
  readonly EOS = 1;
  readonly SEP = 2;

  private vocab: Map<string, number>;
  private idToToken: Map<number, string>;
  private merges: [string, string][];
  private mergeRank: Map<string, number>;

  constructor(data: BPEData) {
    this.vocab = new Map(Object.entries(data.vocab));
    this.idToToken = new Map(
      Object.entries(data.id_to_token).map(([k, v]) => [parseInt(k), v])
    );
    this.merges = data.merges;
    this.mergeRank = new Map(data.merges.map(([a, b], i) => [a + '\x00' + b, i]));
  }

  encode(text: string): number[] {
    const pieces = [...text];
    const nMerges = this.merges.length;
    while (pieces.length >= 2) {
      let bestRank = nMerges;
      let bestI = -1;
      for (let i = 0; i < pieces.length - 1; i++) {
        const rank = this.mergeRank.get(pieces[i] + '\x00' + pieces[i + 1]) ?? nMerges;
        if (rank < bestRank) { bestRank = rank; bestI = i; }
      }
      if (bestI === -1) break;
      pieces[bestI] = pieces[bestI] + pieces[bestI + 1];
      pieces.splice(bestI + 1, 1);
    }
    return pieces.map(p => this.vocab.get(p) ?? this.PAD);
  }

  decodeToken(id: number): string {
    if (id <= 2) return '';
    return this.idToToken.get(id) ?? '';
  }
}
