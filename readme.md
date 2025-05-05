<div align="center">
  <h1>Optimal Samples Selection System User Manual</h1>
  Group 2  
  <br>
  <strong>
    Zhang Ruihao (1220001110), Gao Jiawei (1220011984), Chen Yijia (1220005724), Wang Zheran (1220005681)
  </strong>
  <br><br>
</div>
---

## 1. About

This project presents an **Optimal Samples Selection System**, designed to solve a specific type of grouping problem that arises in sampling, testing, or resource allocation tasks. Given a set of sample candidates and constraints on how they can be grouped, the system aims to select subsets that meet all coverage requirements while minimizing the number of groups used.

We implemented two solving strategies:
- A **Greedy Algorithm**, which provides fast, approximate solutions suitable for large-scale inputs.
- An **Exact Algorithm**, based on constraint programming (CP-SAT solver), which guarantees optimality on smaller instances.

To enhance accessibility and usability, we developed:
- A **desktop GUI** using `Tkinter` for full-feature usage, including .db saving/loading.
- A **mobile-friendly web version** deployed via [Vercel](https://frontend-selector-6gyi.vercel.app/) for lightweight browsing and checking results.

This system supports practical use cases where optimal or near-optimal grouping is required under complex constraints.

-----

## 2. Requirements

Before starting, please ensure you have:

- **Python** (version 3.6 or higher)  
- **Required Python Libraries**  
  - `numpy` – numerical computations and array manipulations  
  - `scipy` – scientific/technical computations (e.g. sparse matrices)  
  - `ortools` – OR‑Tools CP‑SAT solver and other optimization tools  
  - `flask` – To build up back-end UI

-----

## 3. Installation

You can try the following instructions to set the environment.
```bash
conda create -n opm python=3.10
conda activate opm
pip install numpy scipy ortools flask flask-cors
```

---

## 4. File Structure

The system includes the following files:

### **Algorithm Module**: `ai.py`

Contains the core optimization algorithms: `greedy_additive`, `exact_additive`, and utility functions.

### **Main Module**: `main.py`

This Python file integrates both the algorithm module (`ai.py`) and the GUI . It is the entry point for running the GUI locally. It is also responsible for the database management and execution of the algorithm based on user input.

### **Backend Web Module**: `app.py`
This Python file contains the backend logic for running the solver and handling requests (Flask setup in case of external hosting).

### **Frontend Web Interface**: `index.html`

 HTML page for interacting with the system via a web interface.

### **Database Management**: Uses SQLite to save results.

```bash
Optimal-Sample-Selection-System
│
├── backend
│   ├── __pycache__/
│   ├── runs_db/
│   ├── ai.py
│   ├── app.py
│   ├── main.py
│   └── test.py
│
├── frontend/
│
├── readme.md
├── requirements.txt
├── render.yaml
```

---

## 5. How to Execute the System

You have two options to run the system: **locally via the GUI** or **remotely via a web interface**.

### Option 1: Running the GUI Locally

**1.Launch the GUI**:
Open a terminal or command prompt, navigate to the directory containing `main.py`, and run:

```bash
python main.py
```

**Setting Parameters**:

- **m (45~54)**: Total number of samples.
- **n (7~25)**: Number of samples to select.
- **k (4~7)**: Size of each selected sample group.
- **j (s~k)**: Size of each j-combination.
- **s (3~7)**: Size of each s-subset.
- **Threshold**: Minimum number of s-subsets that must be covered by the selected k-combinations.
- **Time Limit**: Set the time limit for the exact solver.

**2.Choosing Sample Mode**:

- **Random**: The system randomly selects the samples.
- **Manual**: You specify the samples manually.

**3.Choosing the Method**:

- **Greedy**: Uses an approximation algorithm.
- **Exact**: Uses the OR-Tools CP-SAT solver for exact optimization.

![image-20250506012732615](D:\Typora\图片\image-20250506012732615.png)



### Option 2: Running via Web Interface

**1.Running the Web Interface Locally**:

Directly visit https://frontend-selector-6gyi.vercel.app/ (if you are in mainland China, please use a VPN)

**2.Setting Parameters**:

- The parameter settings are the same as in *Option 1*

**3.Running the Algorithm**:
After entering the parameters, click the `Run Algorithm` button to send the request to the `backend server`. The server will process the request and return the selected samples and result groups.

**4.Viewing Results**:
The results will be displayed on the webpage, showing the selected samples, groups, and database file name.
![image-20250506012508230](D:\Typora\图片\image-20250506012508230.png)

-----

## 6. System Output

The output includes:

- **Number of Groups Found**: Total number of valid $k$-combinations selected.

- **Execution Time**: Time consumed by the chosen algorithm (Greedy or Exact).

- **Feasibility Status**: Whether the solution meets the threshold coverage requirement.

- **Saved Results**:
  - In the GUI version, each run is automatically saved as a `.db` file under the `runs_db/` folder.
  - The filename encodes the parameter settings, algorithm type, and output index for easy retrieval.

Users can view results by:

- Reloading saved runs from the GUI for inspection.

![image-20250506013251585](D:\Typora\图片\image-20250506013251585.png)

