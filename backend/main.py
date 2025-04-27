import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import os
import random
import time
from itertools import combinations
import threading

# Import algorithm module (assumes ai.py is in the same directory)
from ai import CoverProblem, greedy_additive, exact_additive, mask_to_combo

# -------------------- Database Management --------------------
DB_DIR = os.path.join(os.getcwd(), "runs_db")
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# Save run to SQLite, include samples list
def save_run(params, samples, groups, alg_elapsed):
    m, n, k, j, s, method, thresh, run_id = params
    count = len(groups)
    db_name = f"{m}-{n}-{k}-{j}-{s}-{thresh}-{method}-{run_id}-{count}.db"
    path = os.path.join(DB_DIR, db_name)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    # metadata: samples 和 alg_elapsed
    c.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    c.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
              ('samples', ",".join(str(x) for x in samples)))
    c.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
              ('alg_elapsed', f"{alg_elapsed:.4f}"))
    # results 表
    c.execute("CREATE TABLE IF NOT EXISTS results (group_id INTEGER, samples TEXT)")
    for gid, grp in enumerate(groups, start=1):
        c.execute("INSERT INTO results VALUES (?,?)", (gid, ",".join(grp)))
    conn.commit()
    conn.close()
    return db_name

# List existing run records
def list_runs():
    return [f for f in os.listdir(DB_DIR) if f.endswith('.db')]

# Load specific run, return samples list and groups list
def load_run(db_name):
    path = os.path.join(DB_DIR, db_name)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT value FROM metadata WHERE key='samples'")
    samples = c.fetchone()[0].split(',')
    c.execute("SELECT samples FROM results ORDER BY group_id")
    rows = c.fetchall()
    conn.close()
    groups = [tuple(r[0].split(',')) for r in rows]
    return samples, groups

# -------------------- GUI --------------------
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimal Samples Selection System")
        self.geometry('1000x750')
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabelFrame', background='#f0f0f0', font=('Segoe UI', 12, 'bold'))
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        style.configure('TEntry', font=('Segoe UI', 10))
        style.configure('TRadiobutton', background='#f0f0f0', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10, 'bold'))
        style.configure('TProgressbar', thickness=20)
        self.run_counter = {}
        self.create_widgets()

    def create_widgets(self):
        param_frame = ttk.LabelFrame(self, text="Parameters & Run")
        param_frame.pack(fill='x', padx=15, pady=10)
        for i in range(14): param_frame.grid_columnconfigure(i, weight=1)
        labels = ['m (45-54)', 'n (7-25)', 'k (4-7)', 'j (s-k)', 's (3-7)', 'threshold']
        self.entries = {}
        for idx, lbl in enumerate(labels):
            ttk.Label(param_frame, text=lbl).grid(row=0, column=2*idx, sticky='w', padx=5, pady=5)
            ent = ttk.Entry(param_frame, width=6)
            ent.grid(row=0, column=2*idx+1, sticky='w')
            key = lbl.split()[0]
            self.entries[key] = ent
        self.entries['threshold'].insert(0, '1')
        mode_frame = ttk.LabelFrame(param_frame, text="Sample Mode")
        mode_frame.grid(row=1, column=0, columnspan=6, sticky='w', padx=5, pady=5)
        self.var_mode = tk.StringVar(value='random')
        ttk.Radiobutton(mode_frame, text='Random', variable=self.var_mode,
                        value='random', command=self.toggle_manual).pack(side='left', padx=10)
        ttk.Radiobutton(mode_frame, text='Manual', variable=self.var_mode,
                        value='manual', command=self.toggle_manual).pack(side='left', padx=10)
        self.manual_ent = ttk.Entry(mode_frame, width=30, state='disabled')
        self.manual_ent.pack(side='left', padx=10)
        method_frame = ttk.LabelFrame(param_frame, text="Solve Method")
        method_frame.grid(row=1, column=6, columnspan=6, sticky='w', padx=5, pady=5)
        self.var_method = tk.StringVar(value='greedy')
        ttk.Radiobutton(method_frame, text='Greedy', variable=self.var_method,
                        value='greedy', command=self.toggle_method).pack(side='left', padx=10)
        ttk.Radiobutton(method_frame, text='Exact', variable=self.var_method,
                        value='exact', command=self.toggle_method).pack(side='left', padx=10)
        ttk.Label(method_frame, text='Time Limit (s)').pack(side='left', padx=10)
        self.time_ent = ttk.Entry(method_frame, width=6)
        self.time_ent.insert(0, '60')
        self.time_ent.pack(side='left', padx=5)
        self.toggle_method()
        self.run_button = ttk.Button(param_frame, text="Run Algorithm",
                                     command=self.run_algorithm_threaded)
        self.run_button.grid(row=0, column=12, rowspan=2, sticky='nsew', padx=5, pady=5)
        self.time_label = ttk.Label(param_frame, text="Elapsed: N/A")
        self.time_label.grid(row=2, column=0, columnspan=6, sticky='w', padx=5, pady=5)
        ttk.Label(param_frame, text='Progress').grid(row=2, column=6, sticky='e')
        self.progress = ttk.Progressbar(param_frame, mode='indeterminate')
        self.progress.grid(row=2, column=7, columnspan=6, sticky='ew', padx=5)
        runs_frame = ttk.LabelFrame(self, text="Saved Runs")
        runs_frame.pack(fill='both', expand=True, padx=15, pady=10)
        runs_frame.grid_rowconfigure(0, weight=1)
        runs_frame.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(runs_frame, columns=('file',), show='headings')
        self.tree.heading('file', text='DB File Name')
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb = ttk.Scrollbar(runs_frame, orient='vertical', command=self.tree.yview)
        hsb = ttk.Scrollbar(runs_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        btn_frame = ttk.Frame(runs_frame)
        btn_frame.grid(row=0, column=2, sticky='n', padx=5)
        ttk.Button(btn_frame, text='Load', command=self.load_selected).pack(fill='x', pady=5)
        ttk.Button(btn_frame, text='Delete', command=self.delete_selected).pack(fill='x', pady=5)
        self.refresh_runs()

    def toggle_manual(self):
        if self.var_mode.get() == 'manual':
            self.manual_ent.configure(state='normal')
        else:
            self.manual_ent.delete(0, 'end')
            self.manual_ent.configure(state='disabled')

    def toggle_method(self):
        if self.var_method.get() == 'exact':
            self.time_ent.configure(state='normal')
        else:
            self.time_ent.configure(state='disabled')

    def run_algorithm_threaded(self):
        self.run_button.configure(state='disabled')
        self.progress.start()
        threading.Thread(target=self._run_algorithm, daemon=True).start()

    def _run_algorithm(self):
        # Input validation
        try:
            m = int(self.entries['m'].get()); n = int(self.entries['n'].get())
            k = int(self.entries['k'].get()); j = int(self.entries['j'].get()); s = int(self.entries['s'].get())
            thresh = int(self.entries['threshold'].get())
            if not (45 <= m <= 54 and 7 <= n <= 25 and 4 <= k <= 7 and 3 <= s <= 7 and s <= j <= k):
                raise ValueError
        except ValueError:
            self.after(0, lambda: messagebox.showerror("Input Error", "Invalid parameters."))
            self.after(0, self._end_progress)
            return
        # Prepare samples
        if self.var_mode.get() == 'random':
            samples = random.sample(range(1, m+1), n)
        else:
            try:
                samples = [int(x) for x in self.manual_ent.get().split(',')]
                if len(samples) != n or any(x<1 or x>m for x in samples): raise ValueError
            except ValueError:
                self.after(0, lambda: messagebox.showerror("Input Error", "Manual input must be n values between 1 and m."))
                self.after(0, self._end_progress)
                return
        prob = CoverProblem(n, k, j, s, thresh)
        method = self.var_method.get()
        start = time.time()
        try:
            if method == 'exact':
                time_limit = int(self.time_ent.get())
                chosen = exact_additive(prob, time_limit)
            else:
                chosen = greedy_additive(prob)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Solver Error", str(e)))
            self.after(0, self._end_progress)
            return
        elapsed = time.time() - start
        # Convert groups
        groups = []
        for idx in sorted(chosen):
            nums = tuple(f"{samples[i]:02d}" for i in mask_to_combo(prob.K_masks[idx], n))
            groups.append(nums)
        # Save
        key = (m, n, k, j, s, method, thresh)
        self.run_counter[key] = self.run_counter.get(key, 0) + 1
        db_name = save_run((m, n, k, j, s, method, thresh, self.run_counter[key]), samples, groups)
        # UI update
        def finish():
            self.time_label.config(text=f"Elapsed: {elapsed:.2f} s")
            messagebox.showinfo("Success", f"Run saved as: {db_name}")
            self.refresh_runs()
            self._end_progress()
        self.after(0, finish)

    def _end_progress(self):
        self.progress.stop()
        self.run_button.configure(state='normal')

    def refresh_runs(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for f in list_runs(): self.tree.insert('', 'end', values=(f,))

    def load_selected(self):
        sel = self.tree.selection()
        if not sel: return
        fname = self.tree.item(sel[0])['values'][0]
        samples, groups = load_run(fname)
        win = tk.Toplevel(self)
        win.title(f"Result: {fname}")
        win.geometry('600x450')
        text = tk.Text(win, wrap='none', font=('Consolas', 10))
        vsb = ttk.Scrollbar(win, orient='vertical', command=text.yview)
        hsb = ttk.Scrollbar(win, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        content = f"Samples: {', '.join(samples)}\nNumber of groups: {len(groups)}\nGroups:\n"
        for g in groups: content += ",".join(g)+"\n"
        text.insert('1.0', content)
        text.configure(state='disabled')
        text.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        win.grid_rowconfigure(0, weight=1); win.grid_columnconfigure(0, weight=1)

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel: return
        fname = self.tree.item(sel[0])['values'][0]
        if messagebox.askyesno("Confirm Deletion", f"Delete run file {fname}? This cannot be undone."):
            os.remove(os.path.join(DB_DIR, fname)); self.refresh_runs()

if __name__ == '__main__':
    app = Application(); app.mainloop()
