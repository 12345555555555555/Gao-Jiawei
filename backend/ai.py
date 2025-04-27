#!/usr/bin/env python3
# combo_cover_final.py
# ---------------------------------------------------------
# 解决 (n, k, j, s, threshold) 组合覆盖最小化问题
#  - 累计覆盖模型：每道 j‑组合需有 ≥ threshold 个 s‑子集被至少 1 组选中 k‑组合覆盖
#  - 支持贪心近似 (--greedy) 与 CP‑SAT 精确最优 (--exact)
# ---------------------------------------------------------

import argparse
from itertools import combinations
from typing import List, Tuple, Dict, Set
import random

# ---------------------------------------------------------
# 位掩码工具函数
# ---------------------------------------------------------
def combo_to_mask(combo: Tuple[int, ...]) -> int:
    """元组 → 位掩码"""
    m = 0
    for i in combo:
        m |= 1 << i
    return m


def mask_to_combo(mask: int, n: int) -> Tuple[int, ...]:
    """位掩码 → 元组 (升序)"""
    return tuple(i for i in range(n) if mask & (1 << i))


def generate_masks(n: int, r: int) -> List[int]:
    """返回 nCr 个 r‑子集的掩码列表"""
    return [combo_to_mask(c) for c in combinations(range(n), r)]


def covers(submask: int, supermask: int) -> bool:
    """判断 submask ⊆ supermask"""
    return (submask & supermask) == submask


def submasks_of(mask: int, s: int, n: int) -> List[int]:
    """给定位掩码 mask，生成其中所有大小为 s 的子集掩码"""
    indices = mask_to_combo(mask, n)
    return [combo_to_mask(c) for c in combinations(indices, s)]


# ---------------------------------------------------------
# 问题数据结构
# ---------------------------------------------------------
class CoverProblem:
    def __init__(self, n: int, k: int, j: int, s: int, thresh: int):
        self.n, self.k, self.j, self.s, self.th = n, k, j, s, thresh
        self.K_masks = generate_masks(n, k)            # 所有 k‑组合
        self.J_masks = generate_masks(n, j)            # 所有 j‑组合

        # 每道 j‑组合的所有 s‑子集
        self.S_by_J: List[List[int]] = [
            submasks_of(j_mask, s, n) for j_mask in self.J_masks
        ]
# ✅ 添加：每个 j‑组合的所有 s‑子集（用 Set[int] 表示）
        self.S_by_J_sets: List[List[Set[int]]] = [
            [{i for i in range(n) if (s_mask >> i) & 1} for s_mask in submasks_of(j_mask, s, n)]
            for j_mask in self.J_masks
        ]
        # 预计算 coverage[j][k] = k‑组合覆盖 j‑组合的 s‑子集数量
        self.coverage: List[List[int]] = []
        for j_idx, s_list in enumerate(self.S_by_J):
            row = []
            for k_mask in self.K_masks:
                cnt = sum(covers(s_mask, k_mask) for s_mask in s_list)
                row.append(cnt)
            self.coverage.append(row)


# ---------------------------------------------------------
# 贪心近似：log-factor guarantee
# ---------------------------------------------------------
from typing import Set
from math import exp
from random import shuffle, choice, sample, random
def greedy_additive(prob) -> Set[int]:
    """多次重启 + 位掩码加速 + 最大增益贪心 + 修复步骤 + 局部删点 + 模拟退火"""

    K = len(prob.K_masks)
    J = len(prob.J_masks)

    # 1. 预计算：K 组合的位掩码列表 & 每个 j 上各 s 子集的位掩码
    k_bits_list = prob.K_masks[:]  # int mask
    s_mask_list = [
        [sum(1 << i for i in s_set) for s_set in prob.S_by_J_sets[j_idx]]
        for j_idx in range(J)
    ]

    # 2. 计算 s_bitmask[j][k]：第 k 个组合在第 j 个需求上覆盖哪些 s 子集
    s_bitmask = [[0]*K for _ in range(J)]
    for j in range(J):
        for s_idx, s_mask in enumerate(s_mask_list[j]):
            for k in range(K):
                if (k_bits_list[k] & s_mask) == s_mask:
                    s_bitmask[j][k] |= 1 << s_idx

    # 可行性检查
    def is_feasible(sel: Set[int]) -> bool:
        for j in range(J):
            mask_acc = 0
            for k in sel:
                mask_acc |= s_bitmask[j][k]
            if mask_acc.bit_count() < prob.th:
                return False
        return True

    # 初解构造：最大增益贪心
    def greedy_construct() -> Set[int]:
        uncovered = [prob.th]*J
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
            max_gain = max(g for g,_ in gains)
            best_ks = [k for g,k in gains if g == max_gain]
            k_choice = choice(best_ks)
            sel.add(k_choice)
            for j in range(J):
                uncovered[j] = max(0, uncovered[j] - s_bitmask[j][k_choice].bit_count())
        return sel

    # 修复步骤：保证初解可行
    def repair(sel: Set[int]) -> Set[int]:
        # 计算当前覆盖
        covered = [0]*J
        for j in range(J):
            for k in sel:
                covered[j] |= s_bitmask[j][k]
        # 只保留每个 j 上实际子集位范围
        masks_all = [(1 << len(s_mask_list[j])) - 1 for j in range(J)]

        # 缺口：当 bit_count < th 时，需要继续添加
        while True:
            # 找出最缺的 j
            deficits = [prob.th - covered[j].bit_count() for j in range(J)]
            if all(d <= 0 for d in deficits):
                break
            # 计算每个 k 的修复增益
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
            # 选最大的修复 k
            k_add = max(gains)[1]
            sel.add(k_add)
            for j in range(J):
                covered[j] |= s_bitmask[j][k_add]
        return sel

    # 冗余删点
    def local_delete(sel: Set[int]) -> Set[int]:
        for k in list(sel):
            if is_feasible(sel - {k}):
                sel.remove(k)
        return sel

    # 模拟退火 + 一换一
    def annealing_search(sel: Set[int]) -> Set[int]:
        best = current = sel.copy()
        T = 5.0
        for _ in range(200):
            if len(current) <= 1:
                break
            # 随机去掉一个
            k_out = choice(tuple(current))
            trial = current - {k_out}
            pool = list(set(range(K)) - trial)
            shuffle(pool)
            moved = False
            for k_in in pool:
                cand = trial | {k_in}
                if is_feasible(cand):
                    Δ = len(cand) - len(current)
                    if Δ < 0 or random() < exp(-Δ/(T+1e-9)):
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
    # 重启次数减为 5
    for _ in range(5):
        sol = greedy_construct()
        sol = repair(sol)
        sol = local_delete(sol)
        sol = annealing_search(sol)
        if best_sol is None or len(sol) < len(best_sol):
            best_sol = sol

    return best_sol



def feasible_additive(sel: Set[int], prob: CoverProblem) -> bool:
    """校验累计覆盖可行性"""
    need = [prob.th] * len(prob.J_masks)
    for k in sel:
        for j in range(len(prob.J_masks)):
            need[j] = max(0, need[j] - prob.coverage[j][k])
    return all(v == 0 for v in need)


# ---------------------------------------------------------
# OR‑Tools CP‑SAT 精确最优
# ---------------------------------------------------------
def exact_additive(prob: CoverProblem, time_limit: int = 60) -> Set[int]:
    # ------------------ 新增：计时导入和建模 ------------------
    import time
    t0_build = time.time()
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError("未安装 OR-Tools，请 `pip install ortools` 后重试")
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{k}") for k in range(len(prob.K_masks))]

    # y[j][t] : 第 j 道 j‑组合的第 t 个 s‑子集是否被覆盖
    y: List[List] = []
    for j_idx, s_list in enumerate(prob.S_by_J):
        y_row = [model.NewBoolVar(f"y_{j_idx}_{t}") for t in range(len(s_list))]
        y.append(y_row)
        # 至少 threshold 个 s‑子集被覆盖
        model.Add(sum(y_row) >= prob.th)

        # 若 y=1 ⇒ 至少一个覆盖它的 k 被选
        for t_idx, s_mask in enumerate(s_list):
            k_cover = [
                k for k, k_mask in enumerate(prob.K_masks)
                if covers(s_mask, k_mask)
            ]
            # Big-M 线性化：y ≤ ∑ x_k
            model.Add(sum(x[k] for k in k_cover) >= y_row[t_idx])

    # 目标：最小化 k‑组合数量
    model.Minimize(sum(x))

    # 建模结束
    build_time = time.time() - t0_build

    # 求解阶段
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    t0_solve = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - t0_solve

    # 状态提示（可选）
    if status == cp_model.FEASIBLE:
        print("⚠️ Exact: time limit reached, returning best feasible solution")
    elif status == cp_model.OPTIMAL:
        print("✅ Exact: optimal found")
    else:
        raise RuntimeError(f"CP-SAT failed with status {solver.StatusName(status)}")
 
    chosen = {k for k in range(len(prob.K_masks)) if solver.Value(x[k])}
    # 返回 (组合索引集合, 建模耗时, 求解耗时)
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
    if not feasible_additive(chosen, prob):
        raise AssertionError("得到的解不满足覆盖要求！")

    # ---------- 输出 ----------
    print(f"#Selected = {len(chosen)}\n")
    for k in sorted(chosen):
        print(mask2letters(prob.K_masks[k], prob.n))


if __name__ == "__main__":
    main()
