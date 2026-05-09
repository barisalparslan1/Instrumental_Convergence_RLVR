"""
End-to-end evaluation driver for InstrumentalEval × OLMo-3.

Runs every "execution" cell from the original notebook in order:
  Section 1b — RLVR step discovery
  Section 2  — environment + (optional) HuggingFace login
  Section 3  — load InstrumentalEval benchmark
  Section 4  — assemble prompt conditions
  Section 5  — sanity check (optional, controlled by --sanity-check)
  Section 6  — run primary evaluation across checkpoints
  Section 6b — merge optional task tags from xlsx
  Section 6c — Gemini second judge (optional)
  Section 6d — Inter-Judge Agreement Rate
  Section 6e — paired t-test / Wilcoxon
  Section 7  — residualization, INLP, summary tables, category breakdown

You can run subsets via CLI flags. Everything else lives in `functions.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from functions import (
    build_prompt_conditions,
    build_projected_df,
    bootstrap_ci,
    CATEGORY_ABBR,
    compute_cir,
    compute_ir,
    compute_mean_score,
    discover_rlvr_steps,
    judge_batch_gemini_async,
    load_benchmark_tasks,
    load_checkpoint,
    generate_response,
    judge_response,
    response_cache_path,
    residualize_scores,
    results_path,
    results_to_df,
    run_checkpoint_eval,
    step_sort_key,
    unload_model,
)


# ============================================================
# Section 1b — RLVR step discovery + runtime estimate
# ============================================================

def section_1b_step_discovery(model_size: str) -> list[str]:
    """Discover RLVR step branches and print a runtime estimate."""
    if not config.EVAL_RLVR_STEPS:
        print("EVAL_RLVR_STEPS=False — skipping step discovery.")
        return []

    if config.RLVR_STEPS:
        print(f"RLVR_STEPS already set manually: {config.RLVR_STEPS}")
        rlvr_steps = list(config.RLVR_STEPS)
    else:
        rlvr_steps = discover_rlvr_steps(
            model_size=model_size,
            max_steps=config.MAX_RLVR_STEPS,
            strategy=config.STEP_SAMPLING_STRATEGY,
        )
        print(f"Selected {len(rlvr_steps)} steps: {', '.join(rlvr_steps)}")

    n_named = len(config.CHECKPOINTS)
    n_steps = len(rlvr_steps)
    n_conds = 1 + int(config.RUN_GOAL_NUDGE) + int(config.RUN_ANTI_IC)
    n_total = (n_named + n_steps) * 76 * n_conds
    est_hrs = n_total * 5 / 3600
    print(f"\nRuntime estimate: ~{n_total} calls, ~{est_hrs:.1f} h on A100 40GB")
    if est_hrs > 10:
        print("  ⚠️  Exceeds typical Colab session — consider reducing MAX_RLVR_STEPS.")
    return rlvr_steps


# ============================================================
# Section 5 — single-task sanity check
# ============================================================

def section_5_sanity_check(tasks: list, prompt_conditions: dict,
                           model_size: str, ckpt: str = "SFT") -> None:
    print(f"\n=== Sanity check on {ckpt} ===")
    model, tokenizer, label = load_checkpoint(ckpt, model_size)
    test_task = tasks[0]
    print(f"Task ID  : {test_task['id']}")
    print(f"Category : {test_task['category']}")
    print(f"Prompt   : {str(test_task['user_prompt'])[:200]}\n")

    try:
        response = generate_response(
            model, tokenizer,
            system_prompt=prompt_conditions["base"],
            user_prompt=str(test_task["user_prompt"]),
        )
        print(f"✅ Generation OK — {response[:200]}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        response = "Test response for judge check."

    try:
        verdict = judge_response(
            task_prompt=str(test_task["user_prompt"]),
            model_response=str(response),
            convergence_category=str(test_task["category"]),
        )
        print(f"✅ Judge OK — score={verdict['score']} verdict={verdict['label']}")
        print(f"   Explanation: {verdict['explanation']}")
    except Exception as e:
        print(f"❌ Judge failed: {e}")
    finally:
        unload_model(model, tokenizer)


# ============================================================
# Section 6 — main evaluation loop
# ============================================================

def section_6_run_eval(tasks: list, prompt_conditions: dict,
                       checkpoints_to_run: list[str],
                       model_size: str) -> dict:
    all_results = {}
    for ckpt in checkpoints_to_run:
        print(f"\n{'='*60}\n  Checkpoint: {ckpt}\n{'='*60}")
        results = run_checkpoint_eval(ckpt, tasks, prompt_conditions, model_size)
        all_results[ckpt] = results
    print("\n🎉 All checkpoints evaluated!")
    return all_results


def section_6_load_results(model_size: str) -> dict:
    """Re-load saved per-checkpoint results from OUTPUT_DIR."""
    all_results = {}
    out = Path(config.OUTPUT_DIR)
    for f in sorted(out.glob(f"results_{model_size}_*.json")):
        label = f.stem.replace(f"results_{model_size}_", "")
        all_results[label] = json.load(open(f))
        print(f"Loaded {f.name}: {len(all_results[label])} entries")
    return all_results


def section_6_clean_failed(model_size: str) -> None:
    """Remove items with score=None from saved result files."""
    for f in Path(config.OUTPUT_DIR).glob(f"results_{model_size}_*.json"):
        results = json.load(open(f))
        cleaned = {k: v for k, v in results.items() if v.get("score") is not None}
        removed = len(results) - len(cleaned)
        if removed:
            print(f"{f.name}: removed {removed} failed verdicts")
            json.dump(cleaned, open(f, "w"), indent=2)


# ============================================================
# Section 6b — merge optional task tags
# ============================================================

def section_6b_merge_tags(df: pd.DataFrame,
                          model_size: str,
                          task_tagging_xlsx: str | None) -> pd.DataFrame:
    if not task_tagging_xlsx:
        df["input_type"] = "untagged"
        return df

    print(f"Loading tags from {task_tagging_xlsx} ...")
    df_tags = pd.read_excel(task_tagging_xlsx)
    tag_dict = dict(zip(df_tags["Task ID"], df_tags["input_type"]))
    print(f"  {len(tag_dict)} tags loaded")

    df["input_type"] = df["task_id"].map(tag_dict).fillna("untagged")
    print("DF input_type distribution:")
    print(df["input_type"].value_counts())

    # Patch saved JSON so the two stay in sync
    for f in sorted(Path(config.OUTPUT_DIR).glob(f"results_{model_size}_*.json")):
        results = json.load(open(f))
        for _, item in results.items():
            item["input_type"] = tag_dict.get(item["task_id"], "untagged")
        json.dump(results, open(f, "w"), indent=2)
        print(f"Patched {f.name}")
    return df


# ============================================================
# Section 6c — Gemini second judge
# ============================================================

def section_6c_gemini_judge(tasks: list, prompt_conditions: dict,
                            model_size: str) -> None:
    """Run Gemini on all cached responses to produce a second-judge file."""
    import nest_asyncio
    nest_asyncio.apply()

    out = Path(config.OUTPUT_DIR)
    response_files = sorted(out.glob(f"responses_{model_size}_*.json"))
    print(f"Found {len(response_files)} cached response files.")

    task_lookup = {
        f"{t.get('id', t['file'])}__{cond}": str(t["user_prompt"])
        for t in tasks for cond in prompt_conditions
    }

    for resp_file in response_files:
        label = resp_file.stem.replace(f"responses_{model_size}_", "")
        gemini_path = out / f"results_gemini_{model_size}_{label}.json"

        response_cache = json.load(open(resp_file))
        existing = json.load(open(gemini_path)) if gemini_path.exists() else {}

        keys_to_judge = [k for k in response_cache if k not in existing]
        if not keys_to_judge:
            print(f"  {label}: all already judged by Gemini ({len(existing)} entries)")
            continue

        print(f"  {label}: judging {len(keys_to_judge)} responses with Gemini...")
        judge_items = []
        for key in keys_to_judge:
            cached = response_cache[key]
            judge_items.append({
                "key":            key,
                "task_prompt":    task_lookup.get(key, ""),
                "model_response": str(cached["response"]),
                "category":       cached["category"],
            })

        verdicts = asyncio.run(judge_batch_gemini_async(judge_items))

        for item, verdict in zip(judge_items, verdicts):
            cached = response_cache[item["key"]]
            existing[item["key"]] = {
                "task_id":     cached["task_id"],
                "category":    cached["category"],
                "condition":   cached["condition"],
                "checkpoint":  label,
                "response":    cached["response"],
                "verdict":     verdict["label"],
                "score":       verdict["score"],
                "explanation": verdict["explanation"],
                "converged":   verdict["label"] == "convergence",
            }
        json.dump(existing, open(gemini_path, "w"), indent=2)
        print(f"  ✅ Saved → {gemini_path.name}")
    print("\n✅ All Gemini judging complete.")


# ============================================================
# Section 6d — Inter-Judge Agreement Rate
# ============================================================

def section_6d_iar(model_size: str) -> pd.DataFrame:
    out = Path(config.OUTPUT_DIR)
    response_files = sorted(out.glob(f"responses_{model_size}_*.json"))
    iar_rows = []

    for resp_file in response_files:
        label = resp_file.stem.replace(f"responses_{model_size}_", "")
        gpt_path    = out / f"results_{model_size}_{label}.json"
        gemini_path = out / f"results_gemini_{model_size}_{label}.json"
        if not (gpt_path.exists() and gemini_path.exists()):
            print(f"  Skipping {label}: missing one of the result files")
            continue

        gpt_results    = json.load(open(gpt_path))
        gemini_results = json.load(open(gemini_path))
        common = set(gpt_results) & set(gemini_results)

        valid = 0
        binary_agree = 0
        score_diffs = []
        for k in common:
            g = gpt_results[k].get("score")
            m = gemini_results[k].get("score")
            if g is None or m is None:
                continue
            valid += 1
            if (g >= 0.5) == (m >= 0.5):
                binary_agree += 1
            score_diffs.append(abs(g - m))

        iar_rows.append({
            "Checkpoint":      label,
            "N (valid pairs)": valid,
            "IAR binary (%)":  round(binary_agree / valid * 100, 2) if valid else float("nan"),
            "Mean |Δscore|":   round(sum(score_diffs) / len(score_diffs), 3) if score_diffs else float("nan"),
        })

    iar_df = pd.DataFrame(iar_rows)
    if iar_df.empty:
        return iar_df

    order = ["SFT", "DPO", "RLVR"] + sorted(
        [c for c in iar_df["Checkpoint"] if "step" in c],
        key=step_sort_key,
    )
    iar_df["__order"] = iar_df["Checkpoint"].apply(
        lambda c: order.index(c) if c in order else len(order)
    )
    iar_df = (iar_df.sort_values("__order")
                    .drop(columns="__order")
                    .reset_index(drop=True))

    print("📊 Inter-Judge Agreement Rate (GPT-4o vs Gemini 2.0 Flash)")
    print("=" * 70)
    print(iar_df.to_string(index=False))
    print(f"\n  Average binary IAR: {iar_df['IAR binary (%)'].mean():.2f}%")
    print(f"  Average |Δscore|:   {iar_df['Mean |Δscore|'].mean():.3f}")

    iar_df.to_csv(Path(config.OUTPUT_DIR) / f"IAR_{model_size}.csv", index=False)
    print(f"  Saved → IAR_{model_size}.csv")
    return iar_df


# ============================================================
# Section 6e — paired stats tests
# ============================================================

def section_6e_stats_tests(df: pd.DataFrame) -> None:
    from scipy import stats
    sft  = df[df.checkpoint == "SFT"].set_index("task_id")["score"]
    rlvr = df[df.checkpoint == "RLVR"].set_index("task_id")["score"]
    common = sft.index.intersection(rlvr.index)
    sft, rlvr = sft.loc[common], rlvr.loc[common]
    if len(common) == 0:
        print("No overlap between SFT/RLVR — skipping stats tests.")
        return

    t_stat, p_t = stats.ttest_rel(sft, rlvr)
    w_stat, p_w = stats.wilcoxon(sft, rlvr)
    print(f"Paired t-test:    t={t_stat:.3f}, p={p_t:.4f}")
    print(f"Wilcoxon:         W={w_stat:.3f}, p={p_w:.4f}")
    print(f"Mean difference:  {(sft - rlvr).mean():+.3f}")
    print(f"Cohen's d:        {(sft - rlvr).mean() / (sft - rlvr).std():.3f}")


# ============================================================
# Section 7 — metric tables: residualization, INLP, summaries
# ============================================================

def section_7_metrics(df: pd.DataFrame, model_size: str) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression

    # Residualization
    step_ckpts = sorted(
        [c for c in df["checkpoint"].unique() if "step" in c],
        key=step_sort_key,
    )
    all_checkpoints = ["SFT", "DPO", "RLVR"] + step_ckpts

    df_residualized = residualize_scores(df, checkpoints=all_checkpoints)
    print("── Residualized mean scores (framing-corrected) ──")
    for ckpt in all_checkpoints:
        raw = df[df["checkpoint"] == ckpt]["score"].mean()
        resid = df_residualized[
            df_residualized["checkpoint"] == ckpt
        ]["score_residualized"].mean()
        print(
            f"  {ckpt:18s}  raw={raw:.3f}  corrected={resid:.3f}  "
            f"framing effect={resid - raw:+.3f}"
        )

    # Variance explained by framing per checkpoint
    print("\n── Variance share explained by framing per checkpoint ──")
    print(f"{'Checkpoint':<20s}  {'R²':>8s}  {'1 - R²':>8s}  {'N':>4s}")
    print("-" * 50)
    r2_rows = []
    for ckpt in all_checkpoints:
        sub = df[
            (df.checkpoint == ckpt)
            & (df.score.notna())
            & (df.verdict != "error")
            & (df.condition == "base")
            & (df.input_type.isin(["harmful", "neutral"]))
        ]
        if len(sub) < 5 or sub["input_type"].nunique() < 2:
            continue
        z = (sub["input_type"] == "harmful").astype(float).values.reshape(-1, 1)
        y = sub["score"].values
        reg = LinearRegression().fit(z, y)
        r2 = reg.score(z, y)
        r2_rows.append({
            "Checkpoint": ckpt,
            "R²":         round(r2, 4),
            "1 - R²":     round(1 - r2, 4),
            "N":          len(sub),
        })
        print(f"{ckpt:<20s}  {r2:>8.4f}  {1-r2:>8.4f}  {len(sub):>4d}")

    r2_df = pd.DataFrame(r2_rows)
    if not r2_df.empty:
        r2_df.to_csv(Path(config.OUTPUT_DIR) / f"R2_framing_{model_size}.csv", index=False)
        print(f"\n  Mean R² across checkpoints: {r2_df['R²'].mean():.4f}")

    # 1) Raw IR summary
    summary_rows = []
    for ckpt in df.checkpoint.unique():
        for cond in df.condition.unique():
            ir = compute_ir(df, ckpt, cond)
            lo, hi = bootstrap_ci(df, ckpt, cond)
            summary_rows.append({
                "Checkpoint": ckpt,
                "Condition":  cond,
                "IR (%)":     round(ir, 2),
                "CI lower":   round(lo, 2),
                "CI upper":   round(hi, 2),
            })
    summary_df = pd.DataFrame(summary_rows)
    print("\n📊 Instrumental Rate Summary (Raw)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(Path(config.OUTPUT_DIR) / f"IR_summary_{model_size}.csv", index=False)

    # 2) INLP projected
    df = df.drop(columns=["score_projected", "converged_projected"], errors="ignore")
    df, ckpt_order, _ = build_projected_df(df, condition="base")

    proj_rows = []
    for ckpt in ckpt_order:
        ir_raw = compute_mean_score(df, ckpt, "base", projected=False)
        ir_proj = compute_mean_score(df, ckpt, "base", projected=True)
        proj_rows.append({
            "Checkpoint":             ckpt,
            "Mean score (raw)":       round(ir_raw,  3),
            "Mean score (projected)": round(ir_proj, 3),
            "Difference":             round(ir_proj - ir_raw, 3),
        })
    proj_summary_df = pd.DataFrame(proj_rows)
    print("\n📊 Mean Score: Raw vs INLP Projected")
    print("=" * 60)
    print(proj_summary_df.to_string(index=False))
    proj_summary_df.to_csv(
        Path(config.OUTPUT_DIR) / f"INLP_summary_{model_size}.csv", index=False
    )

    # 3) Key finding
    print("\n── Key finding: SFT → RLVR trend ──")
    sft_raw   = compute_mean_score(df, "SFT",  "base", projected=False)
    rlvr_raw  = compute_mean_score(df, "RLVR", "base", projected=False)
    sft_proj  = compute_mean_score(df, "SFT",  "base", projected=True)
    rlvr_proj = compute_mean_score(df, "RLVR", "base", projected=True)
    print(f"Raw       SFT→RLVR gap: {rlvr_raw  - sft_raw:+.3f}")
    print(f"Projected SFT→RLVR gap: {rlvr_proj - sft_proj:+.3f}")
    if abs(rlvr_proj - sft_proj) > 0.05:
        print("\n✅ Trend SURVIVES — RLVR amplifies convergence beyond input harmfulness")
    else:
        print("\n→ Trend DISAPPEARS — effect explained by input harmfulness")
    return df


# ============================================================
# Section 7b — category breakdown
# ============================================================

def section_7b_category_breakdown(df: pd.DataFrame, model_size: str,
                                  condition: str = "base") -> None:
    cir_rows = []
    for ckpt in df.checkpoint.unique():
        cir = compute_cir(df, ckpt, condition)
        row = {"Checkpoint": ckpt, "Condition": condition}
        for cat, abbr in CATEGORY_ABBR.items():
            row[abbr] = round(cir.get(cat, float("nan")), 2)
        row["IR"] = round(compute_ir(df, ckpt, condition), 2)
        cir_rows.append(row)

    cir_df = pd.DataFrame(cir_rows)
    print("\n📊 Category-Specific Instrumental Rates (%)")
    print(cir_df.to_string(index=False))
    cir_df.to_csv(
        Path(config.OUTPUT_DIR) / f"CIR_breakdown_{model_size}.csv", index=False
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run a single-task sanity check before the full eval.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the model-loading/inference loop and analyze "
                             "previously-saved results.")
    parser.add_argument("--gemini", action="store_true",
                        help="Run the Gemini second judge for IAR analysis.")
    parser.add_argument("--require-gemini", action="store_true",
                        help="Fail at startup if GEMINI_API_KEY is missing.")
    args = parser.parse_args()

    config.assert_required_secrets(require_gemini=args.require_gemini or args.gemini)

    # HuggingFace login (optional)
    if config.HF_TOKEN:
        from huggingface_hub import login
        login(token=config.HF_TOKEN)
        print("✅ HuggingFace login successful.")

    # 1b — step discovery
    rlvr_steps = section_1b_step_discovery(config.MODEL_SIZE)

    # 3 — load benchmark
    tasks = load_benchmark_tasks()

    # 4 — prompt conditions
    prompt_conditions = build_prompt_conditions(
        run_goal_nudge=config.RUN_GOAL_NUDGE,
        run_anti_ic=config.RUN_ANTI_IC,
    )
    print(f"✅ Prompt conditions: {list(prompt_conditions.keys())}")

    # Optional sanity check
    if args.sanity_check and not args.skip_eval:
        section_5_sanity_check(tasks, prompt_conditions, config.MODEL_SIZE)

    # 6 — run main eval
    checkpoints_to_run = list(config.CHECKPOINTS)
    if config.EVAL_RLVR_STEPS:
        checkpoints_to_run += rlvr_steps

    if not args.skip_eval:
        section_6_run_eval(tasks, prompt_conditions, checkpoints_to_run,
                           config.MODEL_SIZE)
        section_6_clean_failed(config.MODEL_SIZE)

    # Re-load results for analysis
    all_results = section_6_load_results(config.MODEL_SIZE)
    if not all_results:
        print("No saved results found — nothing to analyze.")
        return
    df = results_to_df(all_results)

    # 6b — merge tags (optional)
    df = section_6b_merge_tags(df, config.MODEL_SIZE, config.TASK_TAGGING_XLSX)

    # 6c+6d — Gemini second judge + IAR (optional)
    if args.gemini:
        section_6c_gemini_judge(tasks, prompt_conditions, config.MODEL_SIZE)
        section_6d_iar(config.MODEL_SIZE)

    # 6e — paired stats
    section_6e_stats_tests(df)

    # 7 — metrics
    df = section_7_metrics(df, config.MODEL_SIZE)
    section_7b_category_breakdown(df, config.MODEL_SIZE)

    print("\n✅ All done. Summaries saved to:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
