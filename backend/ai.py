#!/usr/bin/env python3
# ---------------------------------------------------------
# Solve the (n, k, j, s, threshold) combination coverage minimization problem
# - Cumulative coverage model: Each j-combination must have ≥ threshold s-subsets covered by at least 1 selected k-combination
# - Support greedy approximation (--greedy) and CP-SAT exact optimization (--exact)
# ---------------------------------------------------------

import argparse
from itertools import combinations
from typing import List, Tuple, Dict, Set
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import random
# ---------------------------------------------------------
# Bitmask utility functions
# ---------------------------------------------------------


def combo_to_mask(combo: Tuple[int, ...]) -> int:
    """tuple → bitmask"""
    m = 0
    for i in combo:
        m |= 1 << i
    return m


def mask_to_combo(mask: int, n: int) -> Tuple[int, ...]:
    """bitmask → tuple (ascending)"""
    return tuple(i for i in range(n) if mask & (1 << i))


def generate_masks(n: int, r: int) -> List[int]:
    """Returns a masked list of nCr r‑subsets"""
    return [combo_to_mask(c) for c in combinations(range(n), r)]


def covers(submask: int, supermask: int) -> bool:
    """Determine submask ⊆ supermask"""
    return (submask & supermask) == submask


# ---------------------------------------------------------
# Optimized submasks_of, using cache to speed up
# ---------------------------------------------------------

@lru_cache(None)
def cached_combinations(indices, s):
    """Cache calculation combinations to avoid repeated calculations"""
    return list(combinations(indices, s))


def submasks_of(mask: int, s: int, n: int) -> List[int]:
    """Given a bit mask mask, generate all subset masks of size s"""
    indices = mask_to_combo(mask, n)
    return [combo_to_mask(c) for c in cached_combinations(tuple(indices), s)]


# ---------------------------------------------------------
# Problem data structure
# ---------------------------------------------------------
class CoverProblem:
    def __init__(self, n: int, k: int, j: int, s: int, thresh: int):
        self.n, self.k, self.j, self.s, self.th = n, k, j, s, thresh
        self.K_masks = generate_masks(n, k)  # All k‑combinations
        self.J_masks = generate_masks(n, j)  # All j‑combinations

        # All s-subsets for each j-combination
        self.S_by_J: List[List[int]] = [
            submasks_of(j_mask, s, n) for j_mask in self.J_masks
        ]
        # all s‑subsets for each j‑combination (represented as Set[int])
        self.S_by_J_sets: List[List[Set[int]]] = [
            [{i for i in range(n) if (s_mask >> i) & 1} for s_mask in submasks_of(j_mask, s, n)]
            for j_mask in self.J_masks
        ]
        # Precompute coverage[j][k] = the number of s-subsets that cover j-combination with k-combination
        self.coverage: List[List[int]] = []
        self.build_coverage()

    def calculate_coverage(self, j_idx, s_list) -> List[int]:
        """Compute the cover of a j-combination"""
        row = []
        for k_mask in self.K_masks:
            cnt = sum(covers(s_mask, k_mask) for s_mask in s_list)
            row.append(cnt)
        return row

    def build_coverage(self):
        """Use parallelization to calculate self.coverage"""
        with ThreadPoolExecutor() as executor:
            self.coverage = list(
                executor.map(lambda j_idx: self.calculate_coverage(j_idx, self.S_by_J[j_idx]), range(self.j)))


# ---------------------------------------------------------
# Greedy approximation: log-factor guarantee
# ---------------------------------------------------------
from typing import Set
from math import exp
from random import shuffle, choice, sample, random
import numpy as np
from scipy.sparse import lil_matrix

def greedy_additive(prob) -> Set[int]:
    """
    Automatically select an algorithm based on the size of n:
- When prob.n <= 10, use multiple restarts + bitmask greedy + repair + local point deletion + simulated annealing.
- When prob.n > 10, use extremely fast greedy vectorized with pure NumPy.
Return the set of selected solutions.
    """
    n = getattr(prob, 'n', None)
    if n is None:
        # If prob does not have the n attribute, it can be inferred from the mask length:
        # Take the highest bit of the first mask + 1 as n
        sample_mask = prob.K_masks[0]
        n = sample_mask.bit_length()
    if n <= 10:
        return _greedy_structured(prob)
    elif n < 23:
        return _greedy_fast_vec(prob)
    else:
        return _greedy_sparse_vec(prob)


def _greedy_structured(prob) -> Set[int]:
    K = len(prob.K_masks)
    J = len(prob.J_masks)
    th = prob.th
    k_bits_list = prob.K_masks[:]  # int mask
    s_mask_list = [
        [sum(1 << i for i in s_set) for s_set in prob.S_by_J_sets[j_idx]]
        for j_idx in range(J)
    ]
    # 1. Precompute s_bitmask
    s_bitmask = [[0] * K for _ in range(J)]
    for j in range(J):
        for s_idx, s_mask in enumerate(s_mask_list[j]):
            for k in range(K):
                if (k_bits_list[k] & s_mask) == s_mask:
                    s_bitmask[j][k] |= 1 << s_idx

    def is_feasible(sel: Set[int]) -> bool:
        for j in range(J):
            mask_acc = 0
            for k in sel:
                mask_acc |= s_bitmask[j][k]
            if mask_acc.bit_count() < th:
                return False
        return True

    def greedy_construct() -> Set[int]:
        uncovered = [th] * J
        sel = set()
        all_k = list(range(K))
        while any(u > 0 for u in uncovered):
            gains = []
            for k in all_k:
                if k in sel: continue
                gain = sum(
                    min(s_bitmask[j][k].bit_count(), uncovered[j])
                    for j in range(J)
                )
                if gain:
                    gains.append((gain, k))
            if not gains:
                break
            max_gain = max(g for g, _ in gains)
            best_ks = [k for g, k in gains if g == max_gain]
            k_choice = choice(best_ks)
            sel.add(k_choice)
            for j in range(J):
                uncovered[j] = max(0, uncovered[j] - s_bitmask[j][k_choice].bit_count())
        return sel

    def repair(sel: Set[int]) -> Set[int]:
        covered = [0] * J
        for j in range(J):
            for k in sel:
                covered[j] |= s_bitmask[j][k]
        masks_all = [(1 << len(s_mask_list[j])) - 1 for j in range(J)]
        while True:
            deficits = [th - covered[j].bit_count() for j in range(J)]
            if all(d <= 0 for d in deficits):
                break
            gains = []
            for k in range(K):
                if k in sel: continue
                gain = 0
                for j in range(J):
                    if deficits[j] > 0:
                        missing = (~covered[j] & masks_all[j])
                        gain += (s_bitmask[j][k] & missing).bit_count()
                if gain:
                    gains.append((gain, k))
            if not gains:
                break
            k_add = max(gains)[1]
            sel.add(k_add)
            for j in range(J):
                covered[j] |= s_bitmask[j][k_add]
        return sel

    def local_delete(sel: Set[int]) -> Set[int]:
        for k in list(sel):
            if is_feasible(sel - {k}):
                sel.remove(k)
        return sel

    def annealing_search(sel: Set[int]) -> Set[int]:
        best = current = sel.copy()
        T = 5.0
        for _ in range(200):
            if len(current) <= 1:
                break
            k_out = choice(tuple(current))
            trial = current - {k_out}
            pool = list(set(range(K)) - trial)
            shuffle(pool)
            moved = False
            for k_in in pool:
                cand = trial | {k_in}
                if is_feasible(cand):
                    d_len = len(cand) - len(current)
                    if d_len < 0 or random() < exp(-d_len / (T + 1e-9)):
                        current = cand
                        if len(cand) < len(best):
                            best = cand
                        moved = True
                        break
            if not moved:
                current = best.copy()
            T *= 0.9
        return best

    best_sol = None
    for _ in range(5):
        sol = greedy_construct()
        sol = repair(sol)
        sol = local_delete(sol)
        sol = annealing_search(sol)
        if best_sol is None or len(sol) < len(best_sol):
            best_sol = sol
    return best_sol


def _greedy_fast_vec(prob) -> Set[int]:
    K = len(prob.K_masks)
    J = len(prob.J_masks)
    th = prob.th
    k_bits_arr = np.array(prob.K_masks, dtype=np.int32)
    S_masks = [
        np.array([sum(1 << i for i in s) for s in prob.S_by_J_sets[j]], dtype=np.int32)
        for j in range(J)
    ]
    cover = np.zeros((J, K), dtype=np.int32)
    for j in range(J):
        masks = S_masks[j][:, None]
        cover[j] = np.sum((k_bits_arr & masks) == masks, axis=0, dtype=np.int32)
    deficits = np.full(J, th, dtype=np.int32)
    selected = []
    while True:
        gains = np.minimum(cover, deficits[:, None]).sum(axis=0)
        best_k = int(np.argmax(gains))
        best_gain = int(gains[best_k])
        if best_gain <= 0:
            break
        selected.append(best_k)
        deficits = np.maximum(deficits - cover[:, best_k], 0)
        if not np.any(deficits > 0):
            break
    return set(selected)
def _greedy_sparse_vec(prob) -> Set[int]:
    K = len(prob.K_masks)
    J = len(prob.J_masks)
    th = prob.th
    k_bits_arr = np.array(prob.K_masks, dtype=np.int64)
    S_masks = [
        [sum(1 << i for i in s) for s in prob.S_by_J_sets[j]]
        for j in range(J)
    ]


    cover = lil_matrix((J, K), dtype=np.int32)
    for j in range(J):
        s_mask_list = S_masks[j]
        for k in range(K):
            cnt = sum((k_bits_arr[k] & s_mask) == s_mask for s_mask in s_mask_list)
            if cnt > 0:
                cover[j, k] = cnt

    deficits = np.full(J, th, dtype=np.int32)
    selected = set()
    remaining = set(range(K))

    while True:
        best_gain = -1
        best_k = None
        for k in remaining:
            gain = 0
            for j in range(J):
                if deficits[j] <= 0:
                    continue
                gain += min(cover[j, k], deficits[j])
            if gain > best_gain:
                best_gain = gain
                best_k = k

        if best_gain <= 0 or best_k is None:
            break

        selected.add(best_k)
        remaining.remove(best_k)
        for j in range(J):
            deficits[j] = max(0, deficits[j] - cover[j, best_k])
        if not np.any(deficits > 0):
            break

    return selected

# ---------------------------------------------------------
# OR‑Tools CP‑SAT 精确最优
# ---------------------------------------------------------
def exact_additive(prob: CoverProblem, time_limit: int = 60) -> Set[int]:
    # ------------------  Timing import and modeling ------------------
    import time
    t0_build = time.time()
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError("OR-Tools is not installed, please `pip install ortools` and try again")

    # Start modeling
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{k}") for k in range(len(prob.K_masks))]

    # The original direct constraint covers each j combination
    for j_idx, s_list in enumerate(prob.S_by_J):
        # Define integer variable: number of s-subsets covered
        s_covered = model.NewIntVar(prob.th, len(s_list), f"s_covered_{j_idx}")

        # For each subset of s, determine whether it is covered
        s_is_covered = []
        for s_mask in s_list:
            covering_k = [x[k] for k, k_mask in enumerate(prob.K_masks) if covers(s_mask, k_mask)]
            # If the current subset s is covered by at least one k, it is covered
            is_cov = model.NewBoolVar("")
            model.Add(sum(covering_k) >= 1).OnlyEnforceIf(is_cov)
            model.Add(sum(covering_k) == 0).OnlyEnforceIf(is_cov.Not())
            s_is_covered.append(is_cov)

        # Constraint on the number of s-subsets covered
        model.Add(s_covered == sum(s_is_covered))
        model.Add(s_covered >= prob.th)

    # Constraint on the number of s-subsets covered
    model.Minimize(sum(x))

    # Modeling is finished
    build_time = time.time() - t0_build

    # Solving phase
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    t0_solve = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - t0_solve

    # Status prompt
    if status == cp_model.FEASIBLE:
        print("⚠️ Exact: time limit reached, returning best feasible solution")
    elif status == cp_model.OPTIMAL:
        print("✅ Exact: optimal found")
    elif status == cp_model.UNKNOWN:
        # Timed out and no feasible solution was found
        print("❌ Exact: time limit reached, no feasible solution found")
        # Return empty solution directly
        return set(), build_time, solve_time
    else:
        # Other abnormal states such as INFEASIBLE
        raise RuntimeError(f"CP-SAT failed with status {solver.StatusName(status)}")

    chosen = {k for k in range(len(prob.K_masks)) if solver.Value(x[k])}
    # Return (combined index set, modeling time, solution time)
    return chosen, build_time, solve_time





# ---------------------------------------------------------
# 字母输出工具
# ---------------------------------------------------------
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def idx2letter(idx: int) -> str:
    return ALPHA[idx] if idx < 26 else f"X{idx}"


def mask2letters(mask: int, n: int) -> Tuple[str, ...]:
    return tuple(idx2letter(i) for i in mask_to_combo(mask, n))


# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Combination Cover Solver (additive model)")
    p = parser.add_argument
    p("--n", type=int, required=True)
    p("--k", type=int, required=True)
    p("--j", type=int, required=True)
    p("--s", type=int, required=True)
    p("--threshold", type=int, default=1,
      help="每道 j‑组合至少需要覆盖的 s‑子集数量")
    p("--exact", action="store_true", help="使用 OR‑Tools 求全局最优")
    p("--time", type=int, default=60, help="CP‑SAT 最大求解秒数")
    p("--seed", type=int, default=0, help="贪心局部搜索随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    prob = CoverProblem(args.n, args.k, args.j, args.s, args.threshold)

    # ---------- 求解 ----------
    if args.exact:
        chosen = exact_additive(prob, args.time)
    else:
        chosen = greedy_additive(prob)

        # ---------- 校验 ----------
        # if not feasible_additive(chosen, prob):
        raise AssertionError("得到的解不满足覆盖要求！")

    # ---------- 输出 ----------
    print(f"#Selected = {len(chosen)}\n")
    for k in sorted(chosen):
        print(mask2letters(prob.K_masks[k], prob.n))


if __name__ == "__main__":
    main()
