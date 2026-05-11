import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from evaluator.trajectory import Trajectory

# All Part-of-Speech Tags (collected only for JSON/NLP mode):
ALL_POS_TAGS = set()


# ---------------------------------------------------------------------------
# JSON-directory mode  (original, NLP-based, web-agent trajectories)
# ---------------------------------------------------------------------------

def process_trajectory(filename, data_dir, output_dir, print_stats=False):
    global ALL_POS_TAGS

    try:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)

        trajectory = Trajectory.from_json(data)
        trajectory.label_verb_noun_pairs(print_stats=print_stats)

        verb_counter = Counter()
        noun_counter = Counter()
        pair_counter = Counter()

        for action in trajectory.actions:
            if action.output_root_verb:
                verb_counter[action.output_root_verb] += 1
            if action.output_root_noun:
                noun_counter[action.output_root_noun] += 1
            for verb, noun in action.output_verb_noun_pairs:
                pair_counter[(verb, noun)] += 1

        try:
            from evaluator.trajectory import get_nlp_model
            nlp = get_nlp_model()

            for action in trajectory.actions:
                text = action.reasoning if action.reasoning else ""
                if text:
                    doc = nlp(text)
                    for token in doc:
                        ALL_POS_TAGS.add(token.pos_)
        except Exception as e:
            if print_stats:
                print(f"Error collecting POS tags: {e}")

        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(trajectory.to_json(), f, indent=2)

        return verb_counter, noun_counter, pair_counter

    except Exception as e:
        if print_stats:
            print(f"Error processing {filename}: {e}")
        return Counter(), Counter(), Counter()


def process_all_trajectories(data_dir, output_dir, print_stats=False):
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} trajectory files")

    total_verb_counter = Counter()
    total_noun_counter = Counter()
    total_pair_counter = Counter()

    for filename in tqdm(json_files, desc="Processing trajectories"):
        verb_counter, noun_counter, pair_counter = process_trajectory(
            filename, data_dir, output_dir, print_stats
        )
        total_verb_counter.update(verb_counter)
        total_noun_counter.update(noun_counter)
        total_pair_counter.update(pair_counter)

    _print_and_save_stats(
        total_verb_counter, total_noun_counter, total_pair_counter,
        output_dir, pos_tags=sorted(ALL_POS_TAGS)
    )


# ---------------------------------------------------------------------------
# Ducc JSONL mode  (tool-call-based extraction, coding-agent trajectories)
# ---------------------------------------------------------------------------

def process_ducc_jsonl(jsonl_path, output_dir, print_stats=False):
    """Process a Ducc/Claude-Code JSONL file with tool-call-based verb-noun extraction.

    Each trajectory is saved as a separate JSON in output_dir so that
    generate_embeddings.py can read them unchanged.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_verb_counter = Counter()
    total_noun_counter = Counter()
    total_pair_counter = Counter()
    n_processed = 0
    n_skipped = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} records in {jsonl_path}")

    for raw in tqdm(lines, desc="Processing ducc records"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            n_skipped += 1
            continue

        data_id = record.get("data_id", f"traj_{n_processed:06d}")

        try:
            trajectory = Trajectory.from_ducc_record(record)
        except Exception as e:
            if print_stats:
                print(f"  [skip] {data_id}: parse error — {e}")
            n_skipped += 1
            continue

        # Tool-call-based extraction (no NLP model needed)
        trajectory.label_verb_noun_from_tool_calls()

        for action in trajectory.actions:
            if action.output_root_verb:
                total_verb_counter[action.output_root_verb] += 1
            if action.output_root_noun:
                total_noun_counter[action.output_root_noun] += 1
            for verb, noun in action.output_verb_noun_pairs:
                total_pair_counter[(verb, noun)] += 1

        # Save per-trajectory JSON (compatible with generate_embeddings.py)
        output_path = os.path.join(output_dir, f"{data_id}.json")
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(trajectory.to_json(), fout, ensure_ascii=False)

        n_processed += 1

        if print_stats and n_processed % 1000 == 0:
            print(f"  {n_processed} processed, {n_skipped} skipped")

    print(f"\nProcessed {n_processed} trajectories ({n_skipped} skipped)")
    _print_and_save_stats(
        total_verb_counter, total_noun_counter, total_pair_counter, output_dir
    )


# ---------------------------------------------------------------------------
# Shared stats helpers
# ---------------------------------------------------------------------------

def _print_and_save_stats(verb_ctr, noun_ctr, pair_ctr, output_dir, pos_tags=None):
    print(f"\nTotal unique verbs:           {len(verb_ctr)}")
    print(f"Total unique nouns:           {len(noun_ctr)}")
    print(f"Total unique verb-noun pairs: {len(pair_ctr)}")

    print("\nTop 15 verbs:")
    for verb, count in verb_ctr.most_common(15):
        print(f"  {verb}: {count}")

    print("\nTop 15 nouns:")
    for noun, count in noun_ctr.most_common(15):
        print(f"  {noun}: {count}")

    print("\nTop 15 verb-noun pairs:")
    for (verb, noun), count in pair_ctr.most_common(15):
        print(f"  {verb} {noun}: {count}")

    stats = {
        "total_verbs":  len(verb_ctr),
        "total_nouns":  len(noun_ctr),
        "total_pairs":  len(pair_ctr),
        "top_verbs":    dict(verb_ctr.most_common(30)),
        "top_nouns":    dict(noun_ctr.most_common(30)),
        "top_pairs": {
            f"{v} {n}": c for (v, n), c in pair_ctr.most_common(30)
        },
    }
    if pos_tags is not None:
        stats["all_pos_tags"] = pos_tags

    stats_file = os.path.join(output_dir, "verb_noun_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics saved to {stats_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract verb-noun pairs from agent trajectories"
    )
    parser.add_argument("--input", "-i",
                        required=True,
                        help="Input: directory of JSON files (json mode) "
                             "or a single JSONL file (ducc_jsonl mode)")
    parser.add_argument("--input-format",
                        choices=["json", "ducc_jsonl"],
                        default="json",
                        help="Input format (default: json)")
    parser.add_argument("--output-dir", "-o",
                        required=True,
                        help="Directory to save labeled trajectory JSON files")
    parser.add_argument("--print-stats", "-p",
                        action="store_true",
                        help="Print verbose processing statistics")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input path does not exist: {args.input}")
        return 1

    if args.input_format == "ducc_jsonl":
        if not os.path.isfile(args.input):
            print(f"Error: --input must be a JSONL file for ducc_jsonl mode")
            return 1
        print(f"Mode: ducc_jsonl (tool-call-based extraction)")
        print(f"Input JSONL: {args.input}")
        print(f"Output dir:  {args.output_dir}")
        process_ducc_jsonl(args.input, args.output_dir, args.print_stats)
    else:
        if not os.path.isdir(args.input):
            print(f"Error: --input must be a directory for json mode")
            return 1
        print(f"Mode: json (NLP-based extraction)")
        print(f"Input dir:  {args.input}")
        print(f"Output dir: {args.output_dir}")
        process_all_trajectories(args.input, args.output_dir, args.print_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# ==================================================
# All Part-of-Speech Tags (json/NLP mode only):
# ==================================================
# ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART
# PRON PROPN PUNCT SCONJ SPACE SYM VERB X
