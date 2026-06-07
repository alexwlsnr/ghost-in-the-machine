# Implementation Plan: Tier 2 (The "Ghost in the Machine")

**Goal:** Port the Z80-μLM architecture to a high-performance WebAssembly/JavaScript environment, ensuring bit-perfect parity with the original Python implementation.

---

## Stage 1: Data Extraction & Serialization (Python)
*Objective: Convert trained PyTorch models into a compact, browser-friendly binary format.*

- [x] **1.1 Binary Weight Packer:** Create a script to iterate through model weights, pack 2-bit weights into 8-bit bytes, and save as a `.bin` file.
- [x] **1.2 Metadata Manifest:** Generate a `.json` file containing:
    - Model architecture (layer sizes).
    - Trigram bucket counts.
    - Charset mapping.
- [x] **1.3 Parity Verification Suite:** Create a script that compares a PyTorch inference pass with a dummy "binary-loaded" pass to ensure the serialization hasn't corrupted the weights.

## Stage 2: The Wasm Compute Kernel (C/Rust $\rightarrow$ Wasm)
*Objective: Implement the heavy mathematical lifting in a high-performance, low-level language.*

- [x] **2.1 Unpacking & MAC Kernel:** Implement the 2-bit unpacking logic and the 16-bit integer Multiply-Accumulate loop.
- [x] **2.2 Layer Dispatcher:** Create a function that iterates through layers and applies the MAC kernel.
- [x] **2.3 Verification Harness:** A JavaScript test runner that feeds known inputs into the Wasm module and compares the output against expected results (calculated in Python).

## Stage 3: The Logic & Encoding Layer (TypeScript)
*Objective: Replicate the trigram hashing and autoregressive loop in JS/TS.*

- [x] **3.1 Trigram Encoder:** Implement the 128-bucket hashing logic in TypeScript.
- [x] **3.2 Inference Orchestrator:** Manage the "loop" where the model processes input $\rightarrow$ generates character $\rightarrow$ appends to context $\rightarrow$ repeats.
- [x] **3.3 Weight/Asset Loader:** An asynchronous loader for `.bin` and `.json` files.

## Stage 4: The Visual Interface (Frontend)
*Objective: Provide a compelling, retro-themed user experience.*

- [ ] **4.1 CRT Canvas Engine:** A `<canvas>` implementation with scanlines, phosphor bloom, and flicker effects to simulate a vintage monitor.
- [ ] **4.2 Interactive Shell:** A command-line style input field that interfaces with the inference loop.
- [ ] **4.3 System "Boot" Sequence:** A stylized loading sequence that makes the user feel like they are powering up a physical Z80 machine.

---

## Definition of Done (Tier 2)
1.  A model trained in Python can be loaded via URL into the browser.
2.  The browser-based model produces the **exact same character sequences** as the Python model for the same input.
3.  Total initial payload is under 50KB.
