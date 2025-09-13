import numpy as np
import matplotlib.pyplot as plt

def _rows_unique(arr: np.ndarray) -> bool:
    # Assumes 2D
    seen = set()
    for row in arr:
        t = tuple(row.tolist())
        if t in seen:
            return False
        seen.add(t)
    return True

def score_fragments_monotone(
    precursor: np.ndarray,
    fragments: np.ndarray,
    *,
    p: float = 1.0,
    baseline: float = 1e-6,
    use_log: bool = False,
    use_weights: bool = True,
    return_details: bool = False,
):
    """
    Monotone diversity potential (always increases when adding a unique fragment).

    Distances:
      - Normalize so precursor becomes uniform and sums to 1: p' = (1/n) * ones.
      - Normalize fragments and losses the same way.
      - For a fragment i, exclude distances to itself and to its own neutral loss.
      - Use Euclidean distance.

    Per-fragment potential (no double counting):
      For each other fragment j != i, take m_ij = min(||f_i - f_j||, ||f_i - loss_j||).
      A_i = baseline
            + k(||f_i - precursor||)
            + k(||f_i - 0||)
            + 1/2 * sum_{j != i} k(m_ij)     # count each unordered pair once

      with k(d) = d^p, p>=1.

    Total score:
      Score = sum_i g(A_i), with g(x) = log(1+x) if use_log else identity.
    """
    precursor = np.asarray(precursor, dtype=float)
    fragments = np.asarray(fragments)

    if precursor.ndim != 1:
        raise ValueError("precursor must be a 1D array")
    if fragments.ndim != 2 or fragments.shape[1] != precursor.shape[0]:
        raise ValueError("fragments must be a 2D array with shape (k, n) matching precursor length")

    # Ensure all fragments are unique (row-unique)
    frag_int = fragments.astype(np.int64, copy=False)
    if not _rows_unique(frag_int):
        raise ValueError("Duplicate fragments detected. Each fragment must be unique.")

    fragments = fragments.astype(float, copy=False)

    n = precursor.shape[0]
    if np.any(precursor <= 0):
        raise ValueError("All precursor entries must be > 0 to normalize to equal entries.")

    # losses: precursor - fragment (per fragment)
    losses = precursor[None, :] - fragments

    # Normalize so precursor becomes uniform and sums to 1: p' = (1/n) * ones
    s = (1.0 / n) / precursor
    precursor_norm = precursor * s
    fragments_norm = fragments * s
    losses_norm = losses * s

    k = fragments_norm.shape[0]
    if k == 0:
        return 0.0 if not return_details else (0.0, {
            "precursor_norm": precursor_norm,
            "fragments_norm": fragments_norm,
            "losses_norm": losses_norm,
            "A": np.array([]),
            "score_per_fragment": np.array([]),
        })

    # Anchors: fragment-to-precursor and fragment-to-zero-loss(=0)
    d_fp = np.linalg.norm(fragments_norm - precursor_norm[None, :], axis=1)  # (k,)
    d_f0 = np.linalg.norm(fragments_norm, axis=1)                             # (k,)

    # Pairwise blocks
    # Fragment-Fragment distances
    diff_ff = fragments_norm[:, None, :] - fragments_norm[None, :, :]         # (k,k,n)
    d_ff = np.linalg.norm(diff_ff, axis=2)                                    # (k,k)

    # Fragment-Loss distances (note: symmetric w.r.t. swapping i<->j)
    diff_fl = fragments_norm[:, None, :] - losses_norm[None, :, :]            # (k,k,n)
    d_fl = np.linalg.norm(diff_fl, axis=2)                                    # (k,k)

    # Mask self and self-loss (exclude i==j and i->loss_i from candidates)
    np.fill_diagonal(d_ff, np.inf)
    np.fill_diagonal(d_fl, np.inf)

    # For each (i,j), take the closer of fragment j or its loss
    m = np.minimum(d_ff, d_fl)                                                # (k,k)

    # Make the pairwise interaction symmetric and exclude diagonal
    m_sym = np.minimum(m, m.T)                                                # (k,k)
    np.fill_diagonal(m_sym, 0.0)

    # Count each unordered pair once; split equally across the two fragments
    pair_row = 0.5 * np.sum(np.power(m_sym, p), axis=1)                       # (k,)

    # Optional weights
    if use_weights:
        frag_sums = np.sum(fragments_norm, axis=1)
        weights = np.where(frag_sums < 0.5, frag_sums, 1.0 - frag_sums)       # in [0, 0.5]
    else:
        weights = None

    A = baseline + np.power(d_fp, p) + np.power(d_f0, p) + pair_row

    score_per_fragment = np.log1p(A) if use_log else A
    if weights is not None:
        score_per_fragment = weights * score_per_fragment

    score = float(np.sum(score_per_fragment))

    if return_details:
        return score, {
            "precursor_norm": precursor_norm,
            "fragments_norm": fragments_norm,
            "losses_norm": losses_norm,
            "A": A,
            "score_per_fragment": score_per_fragment,
        }
    return score

def _sample_unique_fragments(precursor: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    # Generate k unique fragments with 0 <= f_ij <= precursor_j
    n = len(precursor)
    seen = set()
    rows = []
    attempts = 0
    # Upper bound in worst case could be high; while-loop until we have k unique rows
    while len(rows) < k and attempts < k * 20:
        cand = rng.integers(0, precursor + 1)
        t = tuple(int(x) for x in cand.tolist())
        if t not in seen:
            seen.add(t)
            rows.append(cand)
        attempts += 1

    return np.vstack(rows).reshape(-1, n)

def generate_random_case(n, k=None, rng=None, max_val=12):
    rng = np.random.default_rng() if rng is None else rng
    if k is None:
        k = int(rng.integers(1, 10))  # 1..99 fragments
    # Positive precursor entries
    precursor = rng.integers(1, max_val + 1, size=n)
    # Unique fragments with 0 <= f_ij <= precursor_j
    fragments = _sample_unique_fragments(precursor, k, rng)
    return precursor, fragments

def main_demo(num_examples=2000, n=6):
    rng = np.random.default_rng(42)
    scores = []
    for ex in range(1, num_examples + 1):
        precursor, fragments = generate_random_case(n=n, rng=rng)
        score, details = score_fragments_monotone(precursor, fragments, return_details=True)
        scores.append(score)

    # After all examples, plot histogram of scores
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=60, edgecolor="black", alpha=0.8)
    plt.title(f"Monotone diversity potential scores over {num_examples} random cases")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("mono_scores_hist.png", dpi=150)
    plt.show()
    plt.close()
def add_random_fragment(precursor: np.ndarray, fragments: np.ndarray, rng: np.random.Generator):
    # Create a single random fragment with 0 <= f_j <= precursor_j
    # Ensure uniqueness among fragments; regenerate until unique
    existing = {tuple(int(x) for x in row.tolist()) for row in fragments}
    attempts = 0
    while attempts < 20:
        new_frag = rng.integers(0, precursor + 1)
        t = tuple(int(x) for x in new_frag.tolist())
        if t not in existing:
            return np.vstack([fragments, new_frag])
        attempts += 1

def run_monotone_trials(num_trials=5000, n=6, rng_seed=123, bins=60, save_path="mono_diff_hist.png", show=True):
    rng = np.random.default_rng(rng_seed)
    diffs = np.empty(num_trials, dtype=float)
    neg_count = 0
    zero_count = 0
    atol = 1e-12

    for t in range(num_trials):
        precursor, fragments = generate_random_case(n=n, rng=rng)
        old_score, _ = score_fragments_monotone(precursor, fragments, return_details=True)

        fragments2 = add_random_fragment(precursor, fragments, rng)
        # The newly added fragment is the last row
        if fragments2 is None:
            continue
        if fragments2.shape[0] != fragments.shape[0] + 1:
            continue
        
        new_frag = fragments2[-1]
        new_score, _ = score_fragments_monotone(precursor, fragments2, return_details=True)

        diff = new_score - old_score
        diffs[t] = diff
        if diff < -atol:
            neg_count += 1
            print(f"[ALERT] Negative difference at trial {t}: Δ={diff:.6g}")
            print("  Precursor:", precursor)
            print("  Fragments (before):")
            print(fragments)
            print("  Added fragment:", new_frag)
        elif abs(diff) <= atol:
            zero_count += 1
            print(f"[ALERT] Zero difference (≈0) at trial {t}: Δ={diff:.6g}")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(diffs, bins=bins, edgecolor="black", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.title("Monotone score change (new - old) after adding a unique random fragment")
    plt.xlabel("Δscore")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"Trials: {num_trials}")
    print(f"Negative differences: {neg_count}")
    print(f"Zero differences: {zero_count}")
    print(f"Histogram saved to: {save_path}")

if __name__ == "__main__":
    main_demo(num_examples=10000)
    run_monotone_trials(num_trials=5000, n=6, rng_seed=42, bins=60, save_path="mono_diff_hist.png", show=True)
