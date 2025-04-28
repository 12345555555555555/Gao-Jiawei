from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, random, sqlite3, time
from ai import CoverProblem, greedy_additive, exact_additive, mask_to_combo

app = Flask(__name__)
CORS(app)

DB_DIR = os.path.join(os.getcwd(), 'runs_db')
os.makedirs(DB_DIR, exist_ok=True)

@app.route("/run", methods=["POST"])
def run_solver():
    data = request.json
    try:
        m, n, k, j, s = map(int, (data["m"], data["n"], data["k"], data["j"], data["s"]))
        thresh = int(data["thresh"])
        method = data["method"]
        mode = data["mode"]
        samples = (
            random.sample(range(1, m + 1), n)
            if mode == "random"
            else list(map(int, data["samples"]))
        )
        prob = CoverProblem(n, k, j, s, thresh)

        # —— 统一初始化 build_t 和 solve_t —— 
        build_t = 0.0
        solve_t = 0.0

        if method == "exact":
            # Exact 分支：exact_additive 返回 (chosen_set, build_time, solve_time)
            time_limit = int(data.get("time_limit", 60))
            res = exact_additive(prob, time_limit)
            if isinstance(res, tuple):
                chosen, build_t, solve_t = res
            else:
                # 如果老版本只返回集合，就认为 build/solve 都是 0
                chosen = res
        else:
            # Greedy 分支：我们手动计时 solve 阶段
            t0 = time.time()
            chosen = greedy_additive(prob)
            solve_t = time.time() - t0

        # 组装输出结果
        groups = []
        for idx in sorted(chosen):
            mask = prob.K_masks[idx]
            combo = tuple(f"{samples[i]:02d}" for i in mask_to_combo(mask, n))
            groups.append(combo)

        # 保存到 SQLite
        db_name = f"{m}-{n}-{k}-{j}-{s}-{thresh}-{method}-{random.randint(1000,9999)}-{len(groups)}.db"
        db_path = os.path.join(DB_DIR, db_name)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS results (group_id INTEGER, samples TEXT)")
        for i, grp in enumerate(groups, start=1):
            c.execute("INSERT INTO results VALUES (?,?)", (i, ",".join(grp)))
        conn.commit()
        conn.close()

        # 返回 JSON 给前端
        return jsonify({
            "status":    "success",
            "db_name":   db_name,
            "selected":  samples,
            "groups":    groups,
            "build_time": build_t,
            "solve_time": solve_t
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/runs", methods=["GET"])
def list_runs():
    return jsonify([f for f in os.listdir(DB_DIR) if f.endswith(".db")])

@app.route("/load/<db_name>")
def load_run(db_name):
    path = os.path.join(DB_DIR, db_name)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT samples FROM results ORDER BY group_id")
    rows = c.fetchall()
    conn.close()
    groups = [r[0].split(",") for r in rows]
    return jsonify(groups)

@app.route("/")
def serve_index():
    return send_from_directory("../frontend", "index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
