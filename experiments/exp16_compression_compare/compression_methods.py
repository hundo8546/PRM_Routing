"""
Compression methods for Exp 16.

Four implementations:
  A. None          — baseline, no compression
  B. Stopword      — custom rule-based (existing exp910 implementation)
  C. Caveman       — real LLM-based compression (JuliusBrussee/caveman)
  D. LLMLingua     — learned token-importance compression (Jiang et al. 2023)

Caveman uses claude-haiku-4-5-20251001 (cheapest) via ANTHROPIC_API_KEY.
LLMLingua uses microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank locally.

Estimated cost for caveman over 1918 test steps (deduplicated queries):
  ~$2-5 depending on retrieval context length.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ---------------------------------------------------------------------------
# A. No compression (passthrough)
# ---------------------------------------------------------------------------

def compress_none(text: str) -> str:
    return text


# ---------------------------------------------------------------------------
# B. Custom stopword removal (exp910 implementation — for comparison)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","must","can","need","in","on","at","by","for","with","from","to",
    "of","that","this","it","its","i","we","you","he","she","they","them",
    "their","our","my","your","his","her","which","who","what","as","if","so",
    "but","and","or","not","no","also","about","after","all","already","just",
    "more","than","then","there","these","those","up","out","into","onto",
    "upon","very","much","many","such","some","any","each","every","both",
    "between","through","during","before","after","above","below","while",
})

def compress_stopword(text: str) -> str:
    """Rule-based stopword removal. Custom implementation used in exp910."""
    if not text or not text.strip():
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    compressed = []
    for sent in sentences:
        tokens = re.findall(r"\b[\w']+\b|\d+[%$]?|[.!?]", sent)
        kept = []
        for tok in tokens:
            if tok in '.!?':
                if kept and not kept[-1].endswith('.'):
                    kept[-1] = kept[-1].rstrip()
                continue
            if tok.lower() in _STOPWORDS and not tok[0].isupper() and not tok.isdigit():
                continue
            kept.append(tok)
        if kept:
            compressed.append(' '.join(kept))
    return '. '.join(compressed)


# ---------------------------------------------------------------------------
# C. Caveman — real LLM-based compression (JuliusBrussee/caveman)
# ---------------------------------------------------------------------------

_CAVEMAN_PROMPT = """\
Compress the following text. Remove articles, filler words, hedging, and \
redundant phrases. Preserve all numbers, proper nouns, domain terms, and \
technical details exactly. Return ONLY the compressed text — no explanation.

TEXT:
{text}"""

_caveman_client = None

def _get_caveman_client():
    global _caveman_client
    if _caveman_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Required for caveman compression."
            )
        import anthropic
        _caveman_client = anthropic.Anthropic(api_key=api_key)
    return _caveman_client

def compress_caveman(text: str, model: str = "claude-haiku-4-5-20251001",
                     max_retries: int = 3) -> str:
    """
    Compress text using Claude (JuliusBrussee/caveman approach).
    Uses claude-haiku-4-5-20251001 for cost efficiency.
    """
    if not text or not text.strip():
        return text
    client = _get_caveman_client()
    prompt = _CAVEMAN_PROMPT.format(text=text.strip())
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return text  # unreachable

def compress_caveman_batch(texts: list[str],
                           model: str = "claude-haiku-4-5-20251001",
                           max_workers: int = 10) -> tuple[list[str], float]:
    """
    Compress a list of texts in parallel using caveman.
    Returns (compressed_texts, total_cost_usd_estimate).

    Deduplicates identical texts to save API calls.
    """
    # Deduplicate
    unique_texts = list(dict.fromkeys(t for t in texts if t and t.strip()))
    cache: dict[str, str] = {}

    # Haiku pricing (USD / 1M tokens) — approximate
    INPUT_COST_PER_1M  = 0.80
    OUTPUT_COST_PER_1M = 4.00
    total_input_tokens  = 0
    total_output_tokens = 0

    print(f"  [caveman] compressing {len(unique_texts)} unique texts "
          f"({len(texts)} total, {len(texts)-len(unique_texts)} deduped) "
          f"with {max_workers} workers ...")

    client = _get_caveman_client()

    def _compress_one(text: str) -> tuple[str, str, int, int]:
        prompt = _CAVEMAN_PROMPT.format(text=text.strip())
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                compressed = msg.content[0].text.strip()
                in_tok  = msg.usage.input_tokens
                out_tok = msg.usage.output_tokens
                return text, compressed, in_tok, out_tok
            except Exception as e:
                if attempt == 2:
                    print(f"    WARNING: caveman compress failed after retries: {e}")
                    return text, text, 0, 0
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_compress_one, t): t for t in unique_texts}
        for i, fut in enumerate(as_completed(futures), 1):
            orig, comp, in_t, out_t = fut.result()
            cache[orig] = comp
            total_input_tokens  += in_t
            total_output_tokens += out_t
            if i % 50 == 0:
                print(f"    [caveman] {i}/{len(unique_texts)} done ...")

    cost = (total_input_tokens * INPUT_COST_PER_1M +
            total_output_tokens * OUTPUT_COST_PER_1M) / 1_000_000

    compressed_texts = [cache.get(t, t) if t and t.strip() else t for t in texts]
    print(f"  [caveman] done. cost=${cost:.4f}  "
          f"in={total_input_tokens:,} out={total_output_tokens:,} tokens")
    return compressed_texts, cost


# ---------------------------------------------------------------------------
# D. LLMLingua — learned compression (Jiang et al. 2023)
# ---------------------------------------------------------------------------

_llmlingua_compressor = None

def _get_llmlingua_compressor(device: str = "auto"):
    global _llmlingua_compressor
    if _llmlingua_compressor is None:
        try:
            from llmlingua import PromptCompressor
        except ImportError:
            raise ImportError(
                "llmlingua not installed. Run: pip install llmlingua"
            )
        print("  [llmlingua] loading model "
              "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank ...")
        _llmlingua_compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map=device,
        )
        print("  [llmlingua] model loaded.")
    return _llmlingua_compressor

def compress_llmlingua(text: str, rate: float = 0.5,
                       device: str = "auto") -> str:
    """
    Compress text using LLMLingua-2 at the given compression rate.
    rate=0.5 means keep ~50% of tokens.
    """
    if not text or not text.strip():
        return text
    compressor = _get_llmlingua_compressor(device)
    try:
        result = compressor.compress_prompt(
            text.strip(),
            rate=rate,
            force_tokens=['\n', '?', '.'],
        )
        return result['compressed_prompt']
    except Exception as e:
        print(f"    WARNING: llmlingua compress failed: {e}")
        return text

def compress_llmlingua_batch(texts: list[str], rate: float = 0.5,
                              device: str = "auto") -> list[str]:
    """Compress a list of texts with LLMLingua. Processes sequentially (model is local)."""
    compressor = _get_llmlingua_compressor(device)
    results = []
    print(f"  [llmlingua] compressing {len(texts)} texts at rate={rate} ...")
    for i, text in enumerate(texts, 1):
        if not text or not text.strip():
            results.append(text)
            continue
        try:
            result = compressor.compress_prompt(
                text.strip(),
                rate=rate,
                force_tokens=['\n', '?', '.'],
            )
            results.append(result['compressed_prompt'])
        except Exception as e:
            print(f"    WARNING [{i}]: {e}")
            results.append(text)
        if i % 100 == 0:
            print(f"    [llmlingua] {i}/{len(texts)} done ...")
    print(f"  [llmlingua] done.")
    return results


# ---------------------------------------------------------------------------
# Token counting utility
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Approximate token count (chars / 4)."""
    return max(1, len(text) // 4)

def compression_ratio(original: str, compressed: str) -> float:
    """Fraction of tokens saved. 0.4 = 40% reduction."""
    orig_tok = count_tokens(original)
    comp_tok = count_tokens(compressed)
    return 1.0 - (comp_tok / orig_tok)
