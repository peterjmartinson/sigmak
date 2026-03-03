"""Library module: peer comparison report business logic.

All public functions are extracted from
``scripts/generate_peer_comparison_report.py`` so that the CLI layer
(``sigmak.cli.peers``) can call ``run_peer_comparison()`` directly without
going through argparse.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple
import re

from dotenv import load_dotenv

from sigmak.integration import IntegrationError, IntegrationPipeline, RiskAnalysisResult
from sigmak.text_utils import clean_text
from sigmak.downloads.tenk_downloader import (
    TenKDownloader,
    fetch_company_submissions,
    resolve_ticker_to_cik,
)
from sigmak.peer_discovery import PeerDiscoveryService

logger = logging.getLogger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# Business functions (mirrored from script — kept in sync manually)
# ---------------------------------------------------------------------------

def find_html_in_dir(dirpath: Path) -> Path | None:
    if not dirpath.exists():
        return None
    for ext in ("*.htm", "*.html"):
        matches = list(dirpath.glob(ext))
        if matches:
            return matches[0]
    return None


def locate_filing_html(ticker_dir: Path, year: int) -> Path | None:
    """Locate an HTML filing for a ticker directory, tolerant of mis-filed years."""
    exact = find_html_in_dir(ticker_dir / str(year))
    if exact:
        return exact

    candidates: List[Path] = list(ticker_dir.rglob("*.htm")) + list(ticker_dir.rglob("*.html"))
    if not candidates:
        return None

    for c in candidates:
        if str(year) in c.name or str(year) in str(c.parent.name):
            return c

    date_re = re.compile(r"(\d{4})-\d{2}-\d{2}")
    for c in candidates:
        try:
            txt = c.read_text(encoding="utf-8", errors="ignore")[:4096]
        except OSError:
            continue
        m = date_re.search(txt)
        if m and int(m.group(1)) == year:
            return c

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def ensure_filing(downloader: TenKDownloader, ticker: str, year: int) -> Path:
    """Ensure a 10-K HTML exists for ticker/year; download if missing."""
    ticker_dir = Path(downloader.download_dir) / ticker
    html = locate_filing_html(ticker_dir, year)
    if html:
        return html

    try:
        cik = resolve_ticker_to_cik(ticker)
    except ValueError as e:
        raise RuntimeError(f"Ticker resolution failed for {ticker}: {e}")

    try:
        filings = fetch_company_submissions(cik, form_type="10-K")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch submissions for {ticker}: {e}")

    chosen = None
    for f in filings:
        if str(year) == f.filing_date.split("-")[0]:
            chosen = f
            break
    if not chosen:
        if filings:
            chosen = sorted(filings, key=lambda x: x.filing_date, reverse=True)[0]
            logger.info(
                "No exact-year 10-K for %s; falling back to most recent %s",
                ticker,
                chosen.filing_date,
            )
        else:
            raise RuntimeError(f"No 10-K filings found for {ticker}")

    filing_id = downloader.db.insert_filing(chosen)
    try:
        downloader.download_filing(chosen, filing_id)
    except Exception as e:
        raise RuntimeError(f"Download failed for {ticker}: {e}")

    html = find_html_in_dir(ticker_dir)
    if not html:
        raise RuntimeError(
            f"Downloaded filing for {ticker} but no HTML found at {ticker_dir}"
        )
    return html


def validate_cached_result(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that cached JSON has required fields."""
    risks = data.get("risks", [])
    if not risks:
        return True, ""

    for risk in risks:
        if not risk.get("category"):
            return False, "missing risk classification"

    return True, ""


def load_or_analyze_with_cache(
    pipeline: IntegrationPipeline,
    html_path: str,
    ticker: str,
    year: int,
    retrieve_top_k: int = 100,
) -> RiskAnalysisResult:
    """Load cached results or analyze filing fresh."""
    cache_file = Path(f"output/results_{ticker}_{year}.json")

    if cache_file.exists():
        logger.info("Loading cached results for %s %s...", ticker, year)
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            is_valid, reason = validate_cached_result(data)
            if is_valid:
                logger.info("✅ Using cached results for %s %s", ticker, year)
                return RiskAnalysisResult(
                    ticker=data["ticker"],
                    filing_year=data["filing_year"],
                    risks=data["risks"],
                    metadata=data["metadata"],
                )
            else:
                logger.warning(
                    "Cached results for %s %s incomplete (%s); re-analyzing...",
                    ticker,
                    year,
                    reason,
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "Failed to load cache for %s %s: %s; re-analyzing...", ticker, year, e
            )

    logger.info("Analyzing %s %s from %s...", ticker, year, html_path)
    result = pipeline.analyze_filing(
        html_path=html_path,
        ticker=ticker,
        filing_year=year,
        retrieve_top_k=retrieve_top_k,
        rerank=True,
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        f.write(result.to_json())
    logger.info("💾 Cached results to %s", cache_file)

    return result


def filter_boilerplate(result: RiskAnalysisResult) -> RiskAnalysisResult:
    """Filter out BOILERPLATE category risks."""
    original_count = len(result.risks)
    filtered_risks = [
        r for r in result.risks if r.get("category", "").lower() != "boilerplate"
    ]
    filtered_count = original_count - len(filtered_risks)

    if filtered_count > 0:
        logger.info(
            "Filtered %d BOILERPLATE chunks from %s %s",
            filtered_count,
            result.ticker,
            result.filing_year,
        )

    return RiskAnalysisResult(
        ticker=result.ticker,
        filing_year=result.filing_year,
        risks=filtered_risks,
        metadata=result.metadata,
    )


def compute_severity_avg(result: RiskAnalysisResult) -> float:
    vals = [r.get("severity", {}).get("value", 0.0) for r in result.risks]
    return float(mean(vals)) if vals else 0.0


def compute_category_distribution(result: RiskAnalysisResult) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for r in result.risks:
        cat = r.get("category", "UNCATEGORIZED")
        counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def generate_markdown_report(
    target: str,
    year: int,
    results: List[RiskAnalysisResult],
    outpath: Path,
) -> None:
    target_result = next(r for r in results if r.ticker.upper() == target.upper())
    target_result = filter_boilerplate(target_result)

    peer_results = [r for r in results if r.ticker.upper() != target.upper()]
    peer_results = [filter_boilerplate(r) for r in peer_results]

    lines: List[str] = []
    lines.append(f"# Peer Comparison Report — {target} {year}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    target_sev = compute_severity_avg(target_result)
    peer_sevs = [compute_severity_avg(r) for r in peer_results]
    higher = sum(1 for v in peer_sevs if v < target_sev)
    pct = (higher / max(1, len(peer_sevs))) * 100 if peer_sevs else 100.0

    lines.append(f"**Severity average (target):** {target_sev:.3f}")
    lines.append(f"**Severity percentile vs peers:** {pct:.0f}th")
    lines.append("")

    target_dist = compute_category_distribution(target_result)
    lines.append("## Category Distribution (Target)")
    lines.append("")
    for cat, pctv in sorted(target_dist.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- {cat}: {pctv*100:.1f}%")
    lines.append("")

    lines.append("## Compared Companies")
    lines.append("")
    if peer_results:
        for pr in peer_results:
            lines.append(f"- {pr.ticker.upper()}")
    else:
        lines.append("- (no peers analyzed)")
    lines.append("")

    SHARED_THRESHOLD = 0.35

    def tokenize(s: str) -> set[str]:
        return set(re.findall(r"\w+", (s or "").lower()))

    target_risks = [clean_text(r.get("text", "")) for r in target_result.risks]
    peer_risks: List[Tuple[str, str]] = []
    for pr in peer_results:
        for r in pr.risks:
            peer_risks.append((pr.ticker.upper(), clean_text(r.get("text", ""))))

    unique_previews: List[str] = []
    shared_entries: List[Dict[str, Any]] = []

    for t in target_risks:
        t_tokens = tokenize(t)
        best_score = 0.0
        best_peer: str | None = None
        best_peer_text = ""
        for peer_ticker, ptext in peer_risks:
            p_tokens = tokenize(ptext)
            if not t_tokens and not p_tokens:
                score = 1.0
            else:
                inter = len(t_tokens & p_tokens)
                uni = len(t_tokens | p_tokens)
                score = inter / uni if uni else 0.0
            if score > best_score:
                best_score = score
                best_peer = peer_ticker
                best_peer_text = ptext

        if best_score >= SHARED_THRESHOLD:
            shared_entries.append(
                {
                    "target_preview": (t or "").strip()[:200],
                    "peer": best_peer,
                    "peer_preview": (best_peer_text or "").strip()[:200],
                    "score": best_score,
                }
            )
        else:
            unique_previews.append((t or "").strip()[:200])

    lines.append("## Unique Risks (target, preview)")
    lines.append("")
    for u in unique_previews[:10]:
        lines.append(f"- {u}")
    lines.append("")

    lines.append("## Shared Risks (target vs peers)")
    lines.append("")
    if shared_entries:
        for s in sorted(shared_entries, key=lambda x: -x["score"])[:20]:
            lines.append(
                f"- {s['target_preview']}  — shared with **{s['peer']}** (score {s['score']:.3f})"
            )
            if s["peer_preview"]:
                lines.append(f"  - peer snippet: {s['peer_preview']}")
    else:
        lines.append("- (no shared risks detected at current threshold)")
    lines.append("")

    def normalize_text(s: str) -> str:
        return clean_text(s)

    def sentences(text: str) -> List[str]:
        s = re.split(r"(?<=[.!?])\s+", text)
        return [re.sub(r"[^\w\s]", "", x).strip().lower() for x in s if x.strip()]

    def words(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    company_texts = {
        r.ticker.upper(): "\n".join([clean_text(k.get("text", "")) for k in r.risks])
        for r in results
    }

    target_text = company_texts.get(target.upper(), "")

    prior_path = Path(f"output/results_{target}_{year-1}.json")
    novelty_line = "Textual novelty: N/A (prior year not available)"
    if prior_path.exists():
        try:
            prev = json.loads(prior_path.read_text())
            prev_text = "\n".join([r.get("text", "") for r in prev.get("risks", [])])
            cur_sents = set(sentences(target_text))
            prev_sents = set(sentences(prev_text))
            if cur_sents:
                new = cur_sents - prev_sents
                pct_new = (len(new) / len(cur_sents)) * 100.0
                novelty_line = (
                    f"Textual novelty (YoY): {pct_new:.1f}% new sentences "
                    f"({len(new)} of {len(cur_sents)})"
                )
            else:
                novelty_line = "Textual novelty: no sentences detected in current filing"
        except Exception:
            novelty_line = "Textual novelty: error computing novelty"

    lines.append("## Specific Comparisons")
    lines.append("")
    lines.append(f"- **{novelty_line}**")

    all_companies = list(company_texts.keys())
    token_sets = {c: set(words(company_texts[c])) for c in all_companies}

    def jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        uni = len(a | b)
        return inter / uni if uni else 0.0

    peer_scores: List[float] = []
    peer_tokens = [token_sets[c] for c in all_companies if c != target.upper()]

    for pt in peer_tokens:
        peer_scores.append(jaccard(token_sets.get(target.upper(), set()), pt))

    if peer_scores:
        avg_jaccard = sum(peer_scores) / len(peer_scores)
        lines.append(
            f"- **Peer similarity (avg Jaccard):** {avg_jaccard:.3f} — lower means more company-specific"
        )
    else:
        avg_jaccard = 0.0
        lines.append("- **Peer similarity (avg Jaccard):** N/A (no peers analyzed)")

    def paragraphs(text: str) -> List[str]:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        return paras

    stats: Dict[str, Dict[str, int]] = {}
    for c, txt in company_texts.items():
        w = words(txt)
        p = paragraphs(txt)
        stats[c] = {"words": len(w), "paras": len(p)}

    peer_words = [v["words"] for k, v in stats.items() if k != target.upper()]
    peer_paras = [v["paras"] for k, v in stats.items() if k != target.upper()]
    tgt_words = stats.get(target.upper(), {}).get("words", 0)
    tgt_paras = stats.get(target.upper(), {}).get("paras", 0)

    med_words: float = 0.0
    med_paras: float = 0.0
    if peer_words:
        med_words = median(peer_words)
        med_paras = median(peer_paras)
        lines.append(
            f"- **Risk density & volume:** target words={tgt_words}, "
            f"peers median words={med_words}; target paras={tgt_paras}, "
            f"peers median paras={med_paras}"
        )
    else:
        lines.append(
            f"- **Risk density & volume:** target words={tgt_words}, paras={tgt_paras} (no peers)"
        )

    risk_keywords = [
        "litigation", "volatility", "uncertainty", "adverse", "risk",
        "loss", "decline", "regulatory", "compliance",
    ]
    tone: Dict[str, Dict[str, float]] = {}
    for c, txt in company_texts.items():
        w = words(txt)
        cnt = sum(1 for t in w if t in risk_keywords)
        per_k = (cnt / max(1, len(w))) * 1000
        tone[c] = {"count": float(cnt), "per_k": per_k}

    tgt_tone = tone.get(target.upper(), {"count": 0.0, "per_k": 0.0})
    med_perk: float = 0.0
    if peer_words:
        peer_perks = [tone[c]["per_k"] for c in tone if c != target.upper()]
        med_perk = median(peer_perks) if peer_perks else 0.0
        lines.append(
            f"- **Linguistic tone (risk keywords/1k words):** "
            f"target={tgt_tone['per_k']:.2f}, peers median={med_perk:.2f}"
        )
    else:
        lines.append(
            f"- **Linguistic tone (risk keywords/1k words):** target={tgt_tone['per_k']:.2f} (no peers)"
        )

    lines.append("")
    lines.append(
        "*End of specific comparisons — computed with simple token-based metrics "
        "(see ISSUE #82 for desired refinements).*"
    )

    pair_scores: List[float] = []
    bins: List[int] = [0] * 10
    try:
        for t in target_risks:
            t_tokens = tokenize(t)
            for peer_ticker, ptext in peer_risks:
                p_tokens = tokenize(ptext)
                if not t_tokens and not p_tokens:
                    score = 1.0
                else:
                    inter = len(t_tokens & p_tokens)
                    uni = len(t_tokens | p_tokens)
                    score = inter / uni if uni else 0.0
                pair_scores.append(score)

        for s in pair_scores:
            idx = min(int(s * 10), 9)
            bins[idx] += 1

        lines.append("## Jaccard Histogram (target risks vs peer risks)")
        lines.append("")
        if pair_scores:
            total = len(pair_scores)
            max_bar = 40
            for i, count in enumerate(bins):
                lo = i / 10.0
                hi = (i + 1) / 10.0
                barlen = int((count / total) * max_bar)
                bar = "#" * barlen
                lines.append(f"- {lo:.1f}-{hi:.1f}: {count} {' ' if count < 10 else ''}{bar}")
            lines.append("")
        else:
            lines.append("- No pairwise scores available (insufficient data)")
            lines.append("")
    except Exception:
        lines.append("")
        lines.append("- Jaccard histogram: error computing histogram")
        lines.append("")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")

    yaml_path = outpath.with_suffix(".yaml")
    data: Dict[str, Any] = {
        "ticker": target.upper(),
        "year": int(year),
        "summary": {
            "severity_average": float(target_sev),
            "severity_percentile": int(pct),
        },
        "category_distribution": {k: float(v) for k, v in target_dist.items()},
        "compared_companies": [pr.ticker.upper() for pr in peer_results],
        "unique_risks": list(unique_previews),
        "shared_risks": [
            {
                "target_preview": s["target_preview"],
                "peer": s["peer"],
                "peer_preview": s["peer_preview"],
                "score": float(s["score"]),
            }
            for s in shared_entries
        ],
        "specific_comparisons": {
            "textual_novelty": novelty_line,
            "peer_similarity_avg_jaccard": float(avg_jaccard) if peer_scores else None,
            "risk_density": {
                "target_words": int(tgt_words),
                "peers_median_words": float(med_words) if peer_words else None,
                "target_paras": int(tgt_paras),
                "peers_median_paras": float(med_paras) if peer_paras else None,
            },
            "linguistic_tone_per_k": {
                "target": float(tgt_tone.get("per_k", 0.0)),
                "peers_median": float(med_perk) if peer_words else None,
            },
        },
        "jaccard_histogram": {},
    }

    try:
        if pair_scores:
            total_ps = len(pair_scores)
            for i, count in enumerate(bins):
                lo = i / 10.0
                hi = (i + 1) / 10.0
                key = f"{lo:.1f}-{hi:.1f}"
                data["jaccard_histogram"][key] = int(count)
        else:
            data["jaccard_histogram"] = {}
    except Exception:
        data["jaccard_histogram"] = {}

    try:
        import yaml

        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    except Exception:

        def _dump_yaml(obj: Any, indent: int = 0) -> str:
            pad = "  " * indent
            if isinstance(obj, dict):
                out: List[str] = []
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        out.append(f"{pad}{k}:")
                        out.append(_dump_yaml(v, indent + 1))
                    else:
                        if v is None:
                            out.append(f"{pad}{k}: null")
                        elif isinstance(v, bool):
                            out.append(f"{pad}{k}: {str(v).lower()}")
                        elif isinstance(v, (int, float)):
                            out.append(f"{pad}{k}: {v}")
                        else:
                            sval = str(v)
                            if "\n" in sval:
                                block = "\n".join(
                                    [f"{pad}  {line}" for line in sval.splitlines()]
                                )
                                out.append(f"{pad}{k}: |")
                                out.append(block)
                            else:
                                out.append(f"{pad}{k}: {sval}")
                return "\n".join(out)
            elif isinstance(obj, list):
                out2: List[str] = []
                for it in obj:
                    if isinstance(it, (dict, list)):
                        out2.append(f"{pad}- ")
                        out2.append(_dump_yaml(it, indent + 1))
                    else:
                        out2.append(f"{pad}- {it}")
                return "\n".join(out2)
            else:
                return f"{pad}{obj}\n"

        yaml_path.write_text(_dump_yaml(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_peer_comparison(
    ticker: str,
    year: int,
    max_peers: int,
    explicit_peers: List[str] | None,
    db_only: bool,
    db_path: str = "./database/sec_filings.db",
    download_dir: str = "./data/filings",
) -> None:
    """Run a peer comparison report and write Markdown + YAML to ``output/``.

    Parameters
    ----------
    ticker:      Target company ticker symbol.
    year:        Filing year to analyse.
    max_peers:   Maximum number of peers to include in the report.
    explicit_peers:
                 If provided, use these tickers as the peer list instead of
                 auto-discovery.
    db_only:     When True, skip LLM classification and use ChromaDB only.
    db_path:     Path to the SQLite filings database.
    download_dir:
                 Base directory where 10-K HTML files are stored.
    """
    persist_dir = str(Path(db_path).parent)

    downloader = TenKDownloader(db_path=db_path, download_dir=download_dir)
    svc = PeerDiscoveryService(db_path=db_path)

    pipeline = IntegrationPipeline(
        persist_path=persist_dir,
        db_only_classification=db_only,
    )

    target = ticker.upper()

    try:
        target_html = ensure_filing(downloader, target, year)
    except RuntimeError as e:
        logger.error("Failed to ensure filing for target %s: %s", target, e)
        sys.exit(2)

    try:
        target_result = load_or_analyze_with_cache(
            pipeline, str(target_html), target, year
        )
    except IntegrationError as e:
        logger.error("Analysis failed for target %s: %s", target, e)
        sys.exit(3)

    desired = max_peers
    collected: List[RiskAnalysisResult] = []
    attempted: set[str] = set()

    if explicit_peers:
        candidates = [p.upper() for p in explicit_peers if p.upper() != target]
    else:
        candidates = svc.find_peers_for_ticker(target, top_n=max(desired * 3, desired + 6))

    for cand in candidates:
        if len(collected) >= desired:
            break
        cand_u = cand.upper()
        if cand_u == target:
            continue
        if cand_u in attempted:
            continue
        attempted.add(cand_u)

        try:
            html_path = ensure_filing(downloader, cand_u, year)
        except RuntimeError as e:
            logger.warning(
                "Could not obtain filing for peer %s: %s — trying next candidate",
                cand_u,
                e,
            )
            continue

        try:
            res = load_or_analyze_with_cache(pipeline, str(html_path), cand_u, year)
            collected.append(res)
        except IntegrationError as e:
            logger.warning("Analysis failed for peer %s: %s — skipping", cand_u, e)
            continue

    results = [target_result] + collected
    outpath = Path("output") / f"{target}_Peer_Comparison_{year}.md"
    generate_markdown_report(target, year, results, outpath)
    print(f"Wrote report: {outpath}")
