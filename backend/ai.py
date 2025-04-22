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
def greedy_additive(prob: CoverProblem) -> Set[int]:
    uncovered_count = [prob.th] * len(prob.J_masks)    # 每道 j 还差的 s‑子集数
    selected: Set[int] = set()

    while True:
        best_gain, best_k = 0, None
        for k_idx in range(len(prob.K_masks)):
            if k_idx in selected:
                continue
            # 新选此 k 可减少多少“未覆盖 s‑子集”数量
            gain = sum(
                min(prob.coverage[j][k_idx], uncovered_count[j])
                for j in range(len(prob.J_masks))
            )
            if gain > best_gain:
                best_gain, best_k = gain, k_idx

        if best_gain == 0 or best_k is None:
            break  # 再无改进
        selected.add(best_k)
        # 更新剩余缺口
        for j in range(len(prob.J_masks)):
            uncovered_count[j] = max(
                0, uncovered_count[j] - prob.coverage[j][best_k]
            )
        # 全部满足？
        if all(c == 0 for c in uncovered_count):
            break
    return selected


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
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError("未安装 OR‑Tools，请 `pip install ortools` 后重试")

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
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP‑SAT 未找到可行解，请增大时限或检查参数")

    chosen = {k for k in range(len(prob.K_masks)) if solver.Value(x[k])}
    return chosen


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
