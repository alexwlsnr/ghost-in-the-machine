export interface BPEData {
    type?: 'char' | 'bytelevel';
    vocab: Record<string, number>;
    id_to_token: Record<string, string>;
    merges: [string, string][] | string[];
    special?: Record<string, number>;
}
export interface Tokenizer {
    readonly PAD: number;
    readonly EOS: number;
    readonly SEP: number;
    encode(text: string): number[];
    decodeToken(id: number): string;
}
export declare class BPETokenizer implements Tokenizer {
    readonly PAD = 0;
    readonly EOS = 1;
    readonly SEP = 2;
    private vocab;
    private idToToken;
    private merges;
    private mergeRank;
    constructor(data: BPEData);
    encode(text: string): number[];
    decodeToken(id: number): string;
}
export declare class ByteLevelBPETokenizer implements Tokenizer {
    readonly PAD: number;
    readonly EOS: number;
    readonly SEP: number;
    private vocab;
    private idToToken;
    private merges;
    private mergeRank;
    private byteEnc;
    private byteDec;
    private enc;
    constructor(data: BPEData);
    encode(text: string): number[];
    decodeToken(id: number): string;
}
/** Pick the tokenizer implementation from the manifest's tokenizer section. */
export declare function makeTokenizer(data: BPEData): Tokenizer;
//# sourceMappingURL=bpe_tokenizer.d.ts.map