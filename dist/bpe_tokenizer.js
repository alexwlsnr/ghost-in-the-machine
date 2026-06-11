// BPE tokenizer loaded from the model manifest's `tokenizer` section.
// Encode is O(text_len × n_merges) — fast enough for short prompts.
// Decode is O(1) per token via id→string map.
export class BPETokenizer {
    constructor(data) {
        this.PAD = 0;
        this.EOS = 1;
        this.SEP = 2;
        this.vocab = new Map(Object.entries(data.vocab));
        this.idToToken = new Map(Object.entries(data.id_to_token).map(([k, v]) => [parseInt(k), v]));
        this.merges = data.merges;
        this.mergeRank = new Map(data.merges.map(([a, b], i) => [a + '\x00' + b, i]));
    }
    encode(text) {
        const pieces = [...text];
        const nMerges = this.merges.length;
        while (pieces.length >= 2) {
            let bestRank = nMerges;
            let bestI = -1;
            for (let i = 0; i < pieces.length - 1; i++) {
                const rank = this.mergeRank.get(pieces[i] + '\x00' + pieces[i + 1]) ?? nMerges;
                if (rank < bestRank) {
                    bestRank = rank;
                    bestI = i;
                }
            }
            if (bestI === -1)
                break;
            pieces[bestI] = pieces[bestI] + pieces[bestI + 1];
            pieces.splice(bestI + 1, 1);
        }
        return pieces.map(p => this.vocab.get(p) ?? this.PAD);
    }
    decodeToken(id) {
        if (id <= 2)
            return '';
        return this.idToToken.get(id) ?? '';
    }
}
//# sourceMappingURL=bpe_tokenizer.js.map