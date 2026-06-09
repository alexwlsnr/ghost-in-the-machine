#!/usr/bin/env python3
"""
Generate the Ghost in the Machine research paper as a PDF.

Produces docs/ghost_in_the_machine_paper.pdf with charts and academic formatting.
Run: .venv/bin/python3 py/generate_paper.py
"""

import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

# ── Colour palette ────────────────────────────────────────────────────────────
GREEN   = '#33ff33'
DARK_BG = '#0a0a0a'
ACCENT  = '#00ffcc'
MUTED   = '#888888'
RED     = '#ff4444'
BLUE    = '#4488ff'
ORANGE  = '#ffaa33'

PLOT_BG   = '#0d1117'
GRID_COL  = '#1e2530'
AXIS_COL  = '#444444'

def plot_style():
    plt.rcParams.update({
        'figure.facecolor':  PLOT_BG,
        'axes.facecolor':    PLOT_BG,
        'axes.edgecolor':    AXIS_COL,
        'axes.labelcolor':   '#cccccc',
        'text.color':        '#cccccc',
        'xtick.color':       '#888888',
        'ytick.color':       '#888888',
        'grid.color':        GRID_COL,
        'grid.linewidth':    0.5,
        'legend.facecolor':  '#111111',
        'legend.edgecolor':  AXIS_COL,
        'legend.labelcolor': '#cccccc',
        'font.family':       'monospace',
        'font.size':         9,
    })

FIGURES_DIR = Path('docs/paper_figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Training data ─────────────────────────────────────────────────────────────

WISP_CLASSIC = {
    'epochs':    [5,  10,  15,  20,  25,  50],
    'train':     [0.8437, 0.6390, 0.5558, 0.5031, 0.4621, 0.3399],
    'val':       [0.8085, 0.6485, 0.6167, 0.6168, 0.6255, 0.7077],
    'best_val':  0.613,
    'best_epoch': 15,
    'label': 'Wisp fp32',
    'color': BLUE,
}

SHADE_COMPACT = {
    'epochs':    [5,  10,  20,  30,  50],
    'train':     [0.5988, 0.4705, 0.3117, 0.2307, 0.1689],
    'val':       [0.6140, 0.5886, 0.6777, 0.7846, 0.9329],
    'best_val':  0.580,
    'best_epoch': 10,
    'label': 'Shade fp32',
    'color': ORANGE,
}

SPEC512_V11 = {
    'epochs':    [5,   10,   20,   30,   40,   50,   55,   65],
    'train':     [0.6896,0.6291,0.5818,0.5631,0.5521,0.5447,0.5417,0.5363],
    'val':       [0.6335,0.5838,0.5488,0.5343,0.5232,0.5177,0.5194,0.5107],
    'best_val':  0.5107,
    'best_epoch': 65,
    'label': 'Spec512 v1.1',
    'color': ACCENT,
}

SPEC512_V12 = {
    'epochs':    [5,   10,   20,   30,   40,   50,   55],
    'train':     [0.6880,0.6437,0.6116,0.5946,0.5837,0.5752,0.5714],
    'val':       [0.6294,0.5956,0.5713,0.5520,0.5432,0.5358,0.5343],
    'best_val':  0.5337,
    'best_epoch': 53,
    'label': 'Spec512 v1.2 (ongoing)',
    'color': GREEN,
    'dashed': True,
}

WISP_TERNARY = {
    'epochs':    [5,    10,   15,   20,   25,   30,   35,   40,  41],
    'train':     [0.9696,0.8079,0.7341,0.6869,0.6560,0.6334,0.6167,0.6042,0.6018],
    'val':       [0.9506,0.8068,0.7421,0.6981,0.6747,0.6582,0.6406,0.6315,0.6294],
    'best_val':  0.629,
    'best_epoch': 39,
    'label': 'Wisp ternary',
    'color': '#ff66ff',
}

# ── Figure 1: Training curves by model family ─────────────────────────────────

def fig_training_curves():
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(PLOT_BG)

    # Left: val loss comparison across all models
    ax = axes[0]
    for m in [WISP_CLASSIC, SHADE_COMPACT, SPEC512_V11, SPEC512_V12]:
        ls = '--' if m.get('dashed') else '-'
        ax.plot(m['epochs'], m['val'], color=m['color'], linewidth=1.8,
                linestyle=ls, label=m['label'], marker='o', markersize=3)
        ax.axhline(m['best_val'], color=m['color'], linestyle=':', alpha=0.35, linewidth=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss — All Models', pad=8)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0.45, 0.90)

    # Right: ternary vs fp32 Wisp comparison
    ax2 = axes[1]
    ax2.plot(WISP_CLASSIC['epochs'], WISP_CLASSIC['val'],
             color=BLUE, linewidth=1.8, label='Wisp fp32 (val)', marker='o', markersize=3)
    ax2.plot(WISP_TERNARY['epochs'], WISP_TERNARY['val'],
             color='#ff66ff', linewidth=1.8, label='Wisp ternary (val)', marker='s', markersize=3)
    ax2.axhline(WISP_CLASSIC['best_val'], color=BLUE, linestyle=':', alpha=0.5,
                linewidth=1, label=f'fp32 best ({WISP_CLASSIC["best_val"]:.3f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Ternary vs fp32 Wisp', pad=8)
    ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(0.55, 1.05)

    plt.tight_layout(pad=1.5)
    path = FIGURES_DIR / 'training_curves.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=PLOT_BG)
    plt.close()
    return path

# ── Figure 2: Model size vs quality scatter ────────────────────────────────────

def fig_size_quality():
    plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(PLOT_BG)

    models = [
        ('Wisp fp32',       13.4,   0.613,  BLUE,   140),
        ('Wisp 8-bit',       3.7,   0.613,  BLUE,    80),
        ('Wisp 4-bit',       2.0,   0.633,  BLUE,    60),
        ('Shade fp32',       42.0,  0.580,  ORANGE, 140),
        ('Shade 8-bit',      11.3,  0.580,  ORANGE,  80),
        ('Spec512 v1.1 8bit',27.2,  0.511,  ACCENT, 160),
        ('Spec512 v1.2*',   27.2,   0.534,  GREEN,  160),
        ('Wisp ternary*',    1.2,   0.629,  '#ff66ff', 120),
    ]

    for name, size_mb, val_loss, col, ms in models:
        ax.scatter(size_mb, val_loss, color=col, s=ms, alpha=0.85, zorder=5,
                   edgecolors='white', linewidth=0.4)
        offset_x = size_mb * 0.04 + 0.3
        ax.annotate(name, (size_mb, val_loss),
                    xytext=(size_mb + offset_x, val_loss + 0.003),
                    fontsize=7, color='#cccccc', alpha=0.9)

    ax.invert_xaxis()
    ax.set_xlabel('Model Size (MB)  ← smaller is better')
    ax.set_ylabel('Best Val Loss  ↓ lower is better')
    ax.set_title('Model Size vs Quality  (* still training)', pad=8)
    ax.grid(True, alpha=0.4)

    # Annotate Pareto frontier
    ax.annotate('Pareto frontier', xy=(27, 0.514), fontsize=7.5,
                color=ACCENT, style='italic', alpha=0.7)
    ax.plot([27.2, 11.3, 3.7, 2.0, 1.2],
            [0.511, 0.580, 0.613, 0.633, 0.629],
            color=ACCENT, linewidth=0.8, linestyle='--', alpha=0.4)

    plt.tight_layout(pad=1.5)
    path = FIGURES_DIR / 'size_quality.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=PLOT_BG)
    plt.close()
    return path

# ── Figure 3: Quantization impact ─────────────────────────────────────────────

def fig_quantization():
    plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(PLOT_BG)

    # Bar chart: size reduction
    ax = axes[0]
    labels = ['fp32', 'bf16', 'int8', 'int4g\n(grouped)', 'ternary\n(2-bit)', '1-bit\n(Bonsai)']
    bits   = [32, 16, 8, 4, 2, 1.125]
    cols   = [BLUE, ORANGE, ACCENT, GREEN, '#ff66ff', RED]
    sizes  = [100, 50, 25, 12.5, 6.25, 3.5]

    bars = ax.bar(labels, sizes, color=cols, alpha=0.8, edgecolor='#333333')
    ax.set_ylabel('Relative size (%)')
    ax.set_title('Weight Storage by Format\n(relative to fp32)', pad=8)
    ax.grid(True, axis='y', alpha=0.4)
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{s:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#cccccc')

    # Bar chart: params in 100MB
    ax2 = axes[1]
    params = [100/32*8*1e6/4, 100/16*8*1e6/4, 100/8*8*1e6/4,
              100/4*8*1e6/4, 100/2.25*8*1e6/4, 100/1.125*8*1e6/4]
    params_m = [p/1e6 for p in params]
    bars2 = ax2.bar(labels, params_m, color=cols, alpha=0.8, edgecolor='#333333')
    ax2.set_ylabel('Parameters (millions)')
    ax2.set_title('Deployable Params in 100 MB Budget\n(weights only, approx)', pad=8)
    ax2.grid(True, axis='y', alpha=0.4)
    for bar, p in zip(bars2, params_m):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{p:.0f}M', ha='center', va='bottom', fontsize=7.5, color='#cccccc')

    plt.tight_layout(pad=1.5)
    path = FIGURES_DIR / 'quantization.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=PLOT_BG)
    plt.close()
    return path

# ── Figure 4: Architecture timeline ───────────────────────────────────────────

def fig_architecture_timeline():
    plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(PLOT_BG)
    ax.axis('off')

    milestones = [
        (0.0,  'Byte-level\ntokenizer\n(vocab=258)',          BLUE),
        (0.12, 'KV cache\n+ incremental\nforward',            ACCENT),
        (0.24, 'EOS token\n(self-terminate)',                  GREEN),
        (0.36, 'SIMD128\nWasm kernel',                        ORANGE),
        (0.48, 'Modern arch\n(RoPE+SwiGLU\n+RMSNorm)',        ACCENT),
        (0.60, 'int4g grouped\nquantization',                  GREEN),
        (0.72, 'SODA HUMAN\nnormalisation\n+ scenarios',       ORANGE),
        (0.84, 'WebGPU\nengine',                               '#ff66ff'),
        (0.96, 'Ternary\nSTE training',                        RED),
    ]

    y_line = 0.5
    ax.axhline(y_line, color=AXIS_COL, linewidth=1.5, xmin=0.02, xmax=0.98)

    for i, (x, label, col) in enumerate(milestones):
        y_dot = y_line
        ax.scatter([x], [y_dot], color=col, s=120, zorder=5, edgecolors='white', linewidth=0.6)
        y_text = 0.85 if i % 2 == 0 else 0.15
        ax.annotate('', xy=(x, y_dot), xytext=(x, y_text + (0.07 if i%2==0 else -0.07)),
                    arrowprops=dict(arrowstyle='-', color=col, alpha=0.5, lw=0.8))
        ax.text(x, y_text + (0.1 if i%2==0 else -0.1), label,
                ha='center', va='bottom' if i%2==0 else 'top',
                fontsize=7.5, color='#dddddd', multialignment='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#111111', edgecolor=col, alpha=0.7))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 1.3)
    ax.set_title('Project Milestone Timeline', pad=8, color='#cccccc', fontsize=11)

    plt.tight_layout(pad=0.5)
    path = FIGURES_DIR / 'timeline.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=PLOT_BG)
    plt.close()
    return path

# ── Figure 5: Spec512 v1.1 vs v1.2 direct comparison ─────────────────────────

def fig_spec512_comparison():
    plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor(PLOT_BG)

    ax.plot(SPEC512_V11['epochs'], SPEC512_V11['val'],
            color=ACCENT, linewidth=2, label='v1.1 val (SODA unfiltered, 135K)', marker='o', ms=3)
    ax.plot(SPEC512_V12['epochs'], SPEC512_V12['val'],
            color=GREEN, linewidth=2, linestyle='--',
            label='v1.2 val (clean dataset, 132K, ongoing)', marker='s', ms=3)
    ax.plot(SPEC512_V11['epochs'], SPEC512_V11['train'],
            color=ACCENT, linewidth=1, alpha=0.4, linestyle=':')
    ax.plot(SPEC512_V12['epochs'], SPEC512_V12['train'],
            color=GREEN, linewidth=1, alpha=0.4, linestyle=':')

    ax.axhline(SPEC512_V11['best_val'], color=ACCENT, linestyle='--', alpha=0.4, linewidth=0.8,
               label=f'v1.1 best: {SPEC512_V11["best_val"]:.4f} (ep{SPEC512_V11["best_epoch"]})')
    ax.axhline(SPEC512_V12['best_val'], color=GREEN, linestyle='--', alpha=0.4, linewidth=0.8,
               label=f'v1.2 best so far: {SPEC512_V12["best_val"]:.4f} (ep{SPEC512_V12["best_epoch"]})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Spec512 v1.1 vs v1.2 — Cleaner Data Effect', pad=8)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0.50, 0.68)

    plt.tight_layout(pad=1.5)
    path = FIGURES_DIR / 'spec512_comparison.png'
    plt.savefig(path, dpi=160, bbox_inches='tight', facecolor=PLOT_BG)
    plt.close()
    return path

# ── Build PDF ──────────────────────────────────────────────────────────────────

def build_pdf(output_path: str):
    print("Generating figures...")
    fig_timeline  = fig_architecture_timeline()
    fig_curves    = fig_training_curves()
    fig_size_q    = fig_size_quality()
    fig_quant     = fig_quantization()
    fig_spec      = fig_spec512_comparison()
    print("Figures done. Building PDF...")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    W = A4[0] - 4*cm  # usable width

    # Custom styles
    title_s = ParagraphStyle('Title', parent=styles['Title'],
                              fontSize=20, spaceAfter=4, leading=24,
                              alignment=TA_CENTER)
    subtitle_s = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                 fontSize=11, textColor=colors.HexColor('#555555'),
                                 alignment=TA_CENTER, spaceAfter=2)
    authors_s = ParagraphStyle('Authors', parent=styles['Normal'],
                                fontSize=10, alignment=TA_CENTER, spaceAfter=12)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                         fontSize=13, spaceBefore=14, spaceAfter=5,
                         textColor=colors.HexColor('#1a1a2e'), leading=16)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                         fontSize=11, spaceBefore=10, spaceAfter=4,
                         textColor=colors.HexColor('#2d2d5e'))
    body = ParagraphStyle('Body', parent=styles['Normal'],
                           fontSize=9.5, leading=14, spaceAfter=7,
                           alignment=TA_JUSTIFY)
    caption = ParagraphStyle('Caption', parent=styles['Normal'],
                              fontSize=8, textColor=colors.HexColor('#555555'),
                              alignment=TA_CENTER, spaceAfter=10, spaceBefore=3)
    code_s = ParagraphStyle('Code', parent=styles['Code'],
                             fontSize=7.5, backColor=colors.HexColor('#f5f5f5'),
                             spaceAfter=6, spaceBefore=6)

    def img(path, width=None, height=None):
        w = width or W
        return Image(str(path), width=w, height=height or w * 0.42)

    def hr():
        return HRFlowable(width='100%', thickness=0.5,
                          color=colors.HexColor('#dddddd'), spaceAfter=6, spaceBefore=6)

    story = []

    # ── Title page ──────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.2*cm),
        Paragraph("Ghost in the Machine", title_s),
        Paragraph("Training Byte-Level Language Models for In-Browser Deployment", subtitle_s),
        Spacer(1, 0.4*cm),
        Paragraph("Alex Wilson · with Claude Sonnet 4.6", authors_s),
        Paragraph("June 2026", authors_s),
        hr(),
        Spacer(1, 0.3*cm),
    ]

    # ── Abstract ──────────────────────────────────────────────────────────
    story.append(Paragraph("<b>Abstract</b>", h1))
    story.append(Paragraph(
        "We present Ghost in the Machine, a family of byte-level causal language models designed "
        "for deployment entirely within the browser via WebAssembly. Operating under strict "
        "constraints — a 100 MB model budget, WASM compute only, and no server round-trips — we "
        "trained a progression of models from 3.3M to 27.6M parameters, exploring architecture "
        "evolution, data quality strategies, mixed-precision quantization, WebGPU acceleration, "
        "and ternary weight training. Our best model, Spec512 v1.1, achieves a validation loss "
        "of 0.511 on conversational data and produces coherent multi-turn dialogue at 50–200 tokens "
        "per second in-browser. We further validate a ternary training stack ({−1,0,+1} weights via "
        "straight-through estimator) targeting a 2× parameter density improvement over int8 within "
        "the same deployment budget, and situate our approach relative to BitNet b1.58 and the "
        "recently published Bonsai 1-bit 8B model.",
        body))
    story.append(hr())

    # ── 1. Introduction ───────────────────────────────────────────────────
    story.append(Paragraph("1. Introduction", h1))
    story.append(Paragraph(
        "The dominant paradigm for large language model deployment routes every user query through "
        "a remote server. This creates latency, privacy exposure, bandwidth cost, and a hard "
        "dependency on connectivity. For an AI assistant that responds conversationally in "
        "real-time, even 200ms round-trip latency is noticeable.",
        body))
    story.append(Paragraph(
        "Ghost in the Machine explores the alternative: models that run entirely on the user's "
        "device, loaded once into the browser and executing via WebAssembly. The browser is the "
        "universal runtime — available on every phone, laptop, and desktop without installation. "
        "The constraint is severe: models must fit within roughly 100 MB (practical browser memory "
        "budget), tokenize and infer synchronously in JavaScript or Wasm, and produce coherent "
        "multi-turn dialogue at human-readable generation speeds.",
        body))
    story.append(Paragraph(
        "Rather than attempt to compress an existing large model into this window, we train "
        "small models <i>designed</i> for this constraint from the start — byte-level, no external "
        "tokenizer, progressive quantization during training, and a custom Rust/WASM inference "
        "kernel with SIMD128 acceleration.",
        body))

    story.append(img(fig_timeline, width=W, height=W*0.32))
    story.append(Paragraph("Figure 1. Project milestone timeline from initial architecture through ternary weight training.", caption))

    # ── 2. System Design ──────────────────────────────────────────────────
    story.append(Paragraph("2. System Design and Constraints", h1))
    story.append(Paragraph("<b>2.1 Deployment Target</b>", h2))
    story.append(Paragraph(
        "All models target in-browser inference via <code>WebAssembly</code> with an optional "
        "<code>WebGPU</code> acceleration path. The inference stack is a TypeScript orchestrator "
        "calling a Rust-compiled Wasm kernel for compute-intensive operations. Model weights are "
        "fetched as a flat binary file and mapped directly into Wasm linear memory.",
        body))

    story.append(Paragraph("<b>2.2 Tokenization</b>", h2))
    story.append(Paragraph(
        "We use raw byte-level tokenization: vocabulary size 258 (bytes 0–255 plus PAD=256 and "
        "EOS=257). This eliminates any external tokenizer dependency and ensures the model can "
        "handle any UTF-8 input without out-of-vocabulary tokens. The cost is longer sequences — "
        "a 300-character response becomes ~300 tokens — but it aligns cleanly with the WASM "
        "execution model. A SEP token (ASCII SOH, byte 1) separates query from response during "
        "training, teaching the model a clean response zone.",
        body))

    story.append(Paragraph("<b>2.3 Model Family</b>", h2))
    tdata = [
        ['Model', 'Params', 'd_model', 'Layers', 'Heads', 'ctx', 'fp32 size'],
        ['Wisp',  '3.3M',  '256',     '4',       '4',    '64',  '13 MB'],
        ['Shade', '10.9M', '384',     '6',       '6',    '128', '42 MB'],
        ['Spec512','27.6M','512',     '8',       '8',    '1024','105 MB'],
    ]
    t = Table(tdata, colWidths=[2.5*cm, 1.8*cm, 2*cm, 1.8*cm, 1.8*cm, 1.5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d2d5e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f8ff')]),
        ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
        ('PADDING',    (0,0), (-1,-1), 4),
        ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
    ]))
    story += [t, Paragraph("Table 1. Model family overview.", caption)]

    # ── 3. Architecture Evolution ─────────────────────────────────────────
    story.append(Paragraph("3. Architecture Evolution", h1))
    story.append(Paragraph(
        "We began with a standard causal transformer using pre-norm LayerNorm, ReLU FFN, and "
        "learned positional embeddings (the <i>classic</i> architecture). Through ablation "
        "experiments we evaluated modern architecture features:",
        body))
    story.append(Paragraph(
        "<b>RoPE</b> (Rotary Position Embeddings) improved context generalisation on longer "
        "sequences but added training instability on short-context Wisp. <b>SwiGLU</b> FFN "
        "gave ~2–3% val_loss improvement on Shade-scale models with no inference cost change. "
        "<b>RMSNorm</b> reduced training time ~5% with equivalent quality. <b>Weight tying</b> "
        "between embedding and head matrices saved ~3% parameters at no quality cost.",
        body))
    story.append(Paragraph(
        "An important negative result: response loss masking (computing loss only on response "
        "tokens, not query tokens) significantly hurt training on Wisp and Shade. With query "
        "tokens averaging 40% of each sequence, masking them removes half the training signal "
        "on short utterances. We retained masking only for Spec512 where context is long enough "
        "that response tokens dominate.",
        body))

    # ── 4. Data Strategy ──────────────────────────────────────────────────
    story.append(Paragraph("4. Data Strategy", h1))
    story.append(Paragraph(
        "Training data quality proved more impactful than architecture at our parameter scales. "
        "We identified and resolved two major data quality issues.",
        body))
    story.append(Paragraph("<b>4.1 SODA Character Contamination</b>", h2))
    story.append(Paragraph(
        "The SODA dataset (allenai/soda, CC-BY 4.0) is a large social dialogue corpus generated "
        "using GPT-3 from narrative summaries. Each dialogue involves named characters — "
        '"Rajan", "Carlaton", etc. — whose names appear throughout. Models trained naively on '
        "SODA produced responses like <i>HELLO CARLATON!</i> instead of addressing the user. "
        "We applied HUMAN normalisation: a regex-based vocative replacement that substitutes "
        "all character names in addressable positions (sentence starts, after punctuation, before "
        "punctuation) with the literal token HUMAN. This produces the characteristic "
        '"HELLO HUMAN! HOW ARE YOU?" style that became part of Ghost\'s personality.',
        body))
    story.append(Paragraph("<b>4.2 Scenario-Seeded Generation</b>", h2))
    story.append(Paragraph(
        "To supplement SODA, we built a scenario-seeded generation pipeline using gemma4-e4b-distill "
        "as a teacher model. Instead of asking the teacher to respond to pre-written queries, we "
        "describe a conversational scenario and ask the teacher to generate both sides of the "
        "exchange. This produces GHOST/HUMAN-framed dialogues without character contamination. "
        "We generated 50,000 dialogues across 47 scenario types (greetings, emotional support, "
        "opinions, small-talk, jokes, meta) in both single-turn and multi-turn (2–3 exchange) formats.",
        body))
    story.append(Paragraph("<b>4.3 Spec512 v1.2 Dataset Composition</b>", h2))
    story.append(Paragraph(
        "For Spec512 v1.2 we filtered the SODA corpus to emotional, reaction, and small-talk strata "
        "(rejecting transactional, factual, and profanity-heavy dialogues), yielding 80K clean SODA "
        "dialogues combined with 50K scenario-generated dialogues and 5K distilled meta/jokes "
        "pairs — 135K total training items.",
        body))

    # ── 5. Training Results ───────────────────────────────────────────────
    story.append(Paragraph("5. Training Results", h1))
    story.append(img(fig_curves, width=W, height=W*0.4))
    story.append(Paragraph(
        "Figure 2. Left: validation loss across all model variants. Right: ternary Wisp (fp16 scale "
        "groups, 2-bit weights) vs standard fp32 Wisp. Ternary runs ~3–4 epochs behind fp32 due to "
        "STE warm-up cost but converges to a comparable floor.",
        caption))

    story.append(img(fig_spec, width=W*0.85, height=W*0.85*0.55))
    story.append(Paragraph(
        "Figure 3. Spec512 v1.1 vs v1.2. The cleaner, strata-filtered dataset produces faster "
        "early convergence and a lower floor. v1.2 is still training.",
        caption))

    story.append(Paragraph(
        "Key observations: (1) Shade overfits severely after epoch 10 due to limited "
        "training data relative to parameter count — the train/val gap exceeds 0.5 at epoch 50. "
        "(2) Spec512 shows much more gradual overfitting owing to its large context window and "
        "diverse multi-turn training data. (3) Ternary Wisp's train/val gap remains tight (0.028 "
        "at epoch 41) on the 59K dataset, suggesting the zero-weight sparsity provides effective "
        "regularisation.",
        body))

    # ── 6. Quantization ───────────────────────────────────────────────────
    story.append(Paragraph("6. Quantization", h1))
    story.append(img(fig_quant, width=W, height=W*0.37))
    story.append(Paragraph(
        "Figure 4. Left: storage size relative to fp32 for each quantization format. "
        "Right: approximate deployable parameter count in a 100 MB browser budget.",
        caption))

    story.append(img(fig_size_q, width=W*0.85, height=W*0.85*0.6))
    story.append(Paragraph(
        "Figure 5. Model size vs best validation loss. Pareto-optimal models circled. "
        "Spec512 v1.1 at 8-bit (27 MB) dominates the quality dimension; ternary Wisp "
        "(*ongoing) represents the emerging low-size frontier.",
        caption))

    story.append(Paragraph(
        "We implemented four quantization formats: fp32 (baseline), bfloat16 (2× reduction, "
        "no quality loss), int8 (4× reduction, <0.5% quality delta), and int4g (per-group "
        "4-bit quantization, 8× reduction). Int8 proved the best deployment format for Spec512: "
        "27 MB with negligible quality loss. Int4g introduced visible degradation on Wisp-scale "
        "models but was acceptable for Spec512.",
        body))
    story.append(Paragraph(
        "Quantization-aware training (QAT) with an absmean penalty applied every training step "
        "pushed weights toward the quantization grid during training. This improved post-quantization "
        "quality but added ~15% training time overhead. The QAT penalty was omitted for ternary "
        "models since the STE already constrains the weight distribution.",
        body))

    # ── 7. WebGPU Acceleration ────────────────────────────────────────────
    story.append(Paragraph("7. WebGPU Inference Acceleration", h1))
    story.append(Paragraph(
        "The Wasm SIMD kernel achieves ~200–500 tokens/second for Wisp and ~30–50 tokens/second "
        "for Spec512 on modern laptops. For Spec512 this is borderline acceptable; any larger "
        "model would be unusably slow.",
        body))
    story.append(Paragraph(
        "We implemented a WebGPU inference engine that dispatches WGSL compute shaders for each "
        "forward pass operation (embed, LayerNorm, matmul, attention, FFN). Key design choices: "
        "(1) all weight buffers uploaded to VRAM once at model load; (2) KV cache resident on GPU "
        "across tokens; (3) only 258 logit floats transferred back to CPU per generated token. "
        "Bind groups are pre-created at load time, eliminating the dominant per-token JavaScript "
        "overhead (96 × createBindGroup() calls were previously taking ~96ms per token).",
        body))
    story.append(Paragraph(
        "<b>Firefox compatibility:</b> Firefox's wgpu implementation resolves mapAsync() — the "
        "GPU→CPU transfer synchronization primitive — on a fixed 100ms polling interval, making "
        "GPU inference 10× slower than the Wasm fallback on Firefox. We detect this at runtime "
        "and route Firefox to Wasm. Chrome resolves mapAsync in 1–5ms, achieving 50–200 tokens/second "
        "for Spec512 vs ~30 tokens/second in Wasm.",
        body))

    # ── 8. Ternary Weights ────────────────────────────────────────────────
    story.append(Paragraph("8. Ternary Weight Training", h1))
    story.append(Paragraph(
        "Standard quantization applies a discrete weight grid <i>after</i> training, incurring "
        "quality loss. Ternary training from scratch — as in BitNet b1.58 — instead trains with "
        "weights constrained to {−1, 0, +1} via a straight-through estimator (STE), allowing the "
        "optimizer to discover representations that are naturally ternary.",
        body))
    story.append(Paragraph("<b>8.1 Implementation</b>", h2))
    story.append(Paragraph(
        "We implement TernaryLinear using absmean thresholding: scale = mean(|W|), threshold = "
        "0.5 × scale. Weights with |w| < threshold collapse to 0; others become ±scale. "
        "The STE passes gradients through the continuous fp32 weight, allowing standard Adam "
        "optimization. Roughly 50% of weights are zero at any time, providing built-in sparsity "
        "that acts as regularisation. Embeddings, layer norms, and biases remain fp32.",
        body))
    story.append(Paragraph("<b>8.2 Storage Format</b>", h2))
    story.append(Paragraph(
        "We pack 4 ternary codes per byte using 2 bits each (00=−1, 01=0, 10=+1) with one "
        "fp32 scale per tensor, achieving ~2.25 bits per weight including scale overhead. "
        "This gives roughly 370M deployable parameters in a 100 MB budget — compared to 746M "
        "with Bonsai's 1-bit Q1_0_g128 format (1 bit + FP16 scale per 128 weights).",
        body))

    tdata2 = [
        ['Property', 'Our ternary', 'BitNet b1.58', 'Bonsai Q1_0_g128'],
        ['Weight values',    '{−1, 0, +1}',   '{−1, 0, +1}',    '{−1, +1}'],
        ['Scale granularity','per tensor',     'per tensor',     'per 128 weights'],
        ['Scale dtype',      'FP32',           'FP32',           'FP16'],
        ['Bits/weight',      '~2.25',          '~1.625',         '~1.125'],
        ['100MB budget',     '~370M params',   '~520M params',   '~746M params'],
        ['Training',         'from scratch',   'from scratch',   'from scratch'],
        ['Activation quant', 'none',           '8-bit',          'none stated'],
    ]
    t2 = Table(tdata2, colWidths=[3.8*cm, 3.2*cm, 3.2*cm, 3.8*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2d2d5e')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f8ff')]),
        ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
        ('PADDING',    (0,0), (-1,-1), 4),
        ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
    ]))
    story += [t2, Paragraph("Table 2. Ternary approach comparison.", caption)]

    # ── 9. Lessons Learned ────────────────────────────────────────────────
    story.append(Paragraph("9. Key Findings", h1))
    findings = [
        ("<b>Data quality dominates architecture at small scale.</b>",
         "Cleaning SODA (HUMAN normalisation, strata filtering) improved Spec512 val_loss by "
         "~0.015 more than any architectural change. Scenario-seeded generation produced cleaner "
         "GHOST/HUMAN-framed dialogues than filtered corpora alone."),
        ("<b>Response loss masking hurts small models.</b>",
         "Masking query positions in the training target removes ~50% of training signal for "
         "short utterances. Only beneficial when context length is large enough that response "
         "tokens dominate (ctx ≥ 512)."),
        ("<b>Ternary STE converges ~3–4 epochs slower than fp32 but reaches comparable floors.</b>",
         "The STE warm-up cost is real but modest. At epoch 41, ternary Wisp (val=0.629) "
         "is on a trajectory to match fp32 Wisp's 0.613 ceiling, with a tighter "
         "train/val gap suggesting better generalisation on the larger dataset."),
        ("<b>Browser GPU synchronization latency dominates inference timing.</b>",
         "Firefox's 100ms mapAsync polling turns a theoretically 5× faster GPU path into a "
         "10× slower one. Chrome's 1–5ms resolution is the real target; WebGPU is "
         "a genuine win only on Chrome/Edge today."),
        ("<b>For 1-bit (Bonsai-style) models, BPE tokenization becomes essential.</b>",
         "At 1-bit precision, the model cannot afford to spend weight capacity learning "
         "byte-level spelling mechanics. BPE compression (3–4× shorter sequences) is "
         "the key enabler for 1-bit quality at our scale."),
    ]
    for title, text in findings:
        story.append(Paragraph(f"{title} {text}", body))

    # ── 10. Future Work ───────────────────────────────────────────────────
    story.append(Paragraph("10. Future Work", h1))
    story.append(Paragraph(
        "<b>Ternary Shade.</b> Validate ternary training at 11M parameter scale. If ternary "
        "matches fp32 Shade quality, proceed to Revenant.",
        body))
    story.append(Paragraph(
        "<b>BPE tokenization.</b> A small BPE vocabulary (512–1024 tokens) reduces average "
        "sequence length 3–4×. Combined with ternary weights, this is the core enabler for "
        "Revenant — a 300–700M parameter model in the 100 MB budget.",
        body))
    story.append(Paragraph(
        "<b>Revenant (1-bit, BPE, 300–700M params).</b> A Bonsai-inspired model "
        "trained from scratch with BPE tokenization and 1-bit Q1_0_g128-style weights. "
        "Target: coherent multi-turn conversation in-browser at the quality level of "
        "current 7B server-side models.",
        body))
    story.append(Paragraph(
        "<b>WebGPU activation quantization.</b> Currently only weights are quantized; "
        "activations remain fp32. Adding 8-bit activation quantization (as in BitNet b1.58) "
        "would reduce GPU memory bandwidth further and improve token throughput.",
        body))

    story.append(hr())
    story.append(Paragraph(
        "<i>All models, training code, inference kernels, and the browser demo are "
        "open-source at github.com/alexwlsnr/ghost-in-the-machine.</i>",
        ParagraphStyle('footer', parent=styles['Normal'],
                       fontSize=8.5, alignment=TA_CENTER,
                       textColor=colors.HexColor('#777777'))))

    doc.build(story)
    print(f"PDF written → {output_path}")


if __name__ == '__main__':
    os.makedirs('docs', exist_ok=True)
    build_pdf('docs/ghost_in_the_machine_paper.pdf')
