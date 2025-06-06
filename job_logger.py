import os
import sys
import datetime
import subprocess
import traceback
import filelock  # pip install filelock
from typing import Optional, List, Dict
from pathlib import Path

import logging

# ---------------------------------------------------------------------
# Globals for file paths

STAT_DIR = "stat"
os.makedirs(STAT_DIR, exist_ok=True)

BEST_FAIL_TRACKER_FILE = os.path.join(STAT_DIR, "best_fail_tracker.csv")

# List all models
ALL_MODELS = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B",
    "meta-llama/Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-14B-Base",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "mistralai/Mistral-Nemo-Base-2407",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-Small-24B-Base-2501",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]

# List seeds you care about
ALL_SEEDS = [777, 1234, 308, 4649, 713]  
# List fewshot
ALL_FEWSHOTS = [0, 1, 2, 5, 10]

# ---------------------------------------------------------------------
def determine_tasks_category(tasks: List[str]) -> str:
    """
    Return "mmlu" if "mmlu" is in the list,
    otherwise "others".
    If the user passes multiple tasks that include 'mmlu',
    we label it 'mmlu' for global-tracking purposes.
    """
    if "mmlu" in tasks:
        return "mmlu"
    else:
        return "others"

# ---------------------------------------------------------------------
def parse_shell_script_commands(script_path: str) -> List[str]:
    """
    Read the shell script lines containing `python ... main.py`.
    Return them in the same order as they appear in the script.
    For example, if script has 2 lines calling main.py, 
    we return a list with those 2 lines.
    """
    lines = []
    if not os.path.exists(script_path):
        return lines
    
    with open(script_path, "r", encoding="utf-8") as f:
        for row in f:
            row_strip = row.strip()
            if row_strip.startswith("python ") and "main.py" in row_strip:
                lines.append(row_strip)
    return lines

# ---------------------------------------------------------------------
def match_command_line(full_cmd: str, script_commands: List[str]) -> int:
    """
    Attempt to find the index (0-based) in `script_commands` that
    best matches the current `full_cmd`.
    
    full_cmd might be something like:
      'python /home/work/home/pjm/snullm_eval/main.py --model ...'
    while script_commands might have
      'python /home/work/home/pjm/snullm_eval/main.py --model ...'
    We'll do a simple string match ignoring differences in spacing.
    
    Return -1 if not found.
    """
    # We'll do naive approach: remove repeated spaces, then compare
    def norm(s: str) -> str:
        return " ".join(s.split())
    
    full_cmd_norm = norm(full_cmd)
    for i, scmd in enumerate(script_commands):
        if norm(scmd) == full_cmd_norm:
            return i
    return -1
        
# ---------------------------------------------------------------------

def init_best_fail_tracker():
    """
    If not present, create best_fail_tracker.csv with header:
    Model,Tasks,Fewshot,Seed,Max_Success_BSZ,Min_Fail_BSZ
    """
    if os.path.exists(BEST_FAIL_TRACKER_FILE):
        return
    header = "Model,Tasks,Fewshot,Seed,Max_Success_BSZ,Min_Fail_BSZ"
    with open(BEST_FAIL_TRACKER_FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n")

def read_best_fail_tracker() -> List[str]:
    if not os.path.exists(BEST_FAIL_TRACKER_FILE):
        init_best_fail_tracker()
    with open(BEST_FAIL_TRACKER_FILE, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f.readlines()]

def write_best_fail_tracker(lines: List[str]):
    with open(BEST_FAIL_TRACKER_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def update_best_fail_tracker(
    model: str,
    tasks_cat: str,   # "mmlu" or "others"
    fewshot: int,
    seed: int,
    bsz: int,
    success: bool,
    is_oom_error: bool = False
):
    """
    If success=True, then Max_Success_BSZ may be updated if bsz is bigger.
    If success=False AND is_oom_error=True, then Min_Fail_BSZ may be updated if bsz is smaller.
    If success=False AND is_oom_error=False, we don't update Min_Fail_BSZ.
    We'll search for an existing row (model, tasks_cat, fewshot, seed).
    If none, create it.
    """
    with filelock.FileLock(BEST_FAIL_TRACKER_FILE + ".lock"):
        lines = read_best_fail_tracker()
        # lines[0] is header
        found_idx = -1
        for i in range(1, len(lines)):
            row = lines[i].split(",")
            if len(row) < 6:
                continue
            # Model,Tasks,Fewshot,Seed,Max_Success_BSZ,Min_Fail_BSZ
            if (row[0] == model and
                row[1] == tasks_cat and
                row[2] == str(fewshot) and
                row[3] == str(seed)):
                found_idx = i
                break
        
        if found_idx == -1:
            # create
            max_succ = bsz if success else ""
            min_fail = bsz if (not success and is_oom_error) else ""
            new_line = f"{model},{tasks_cat},{fewshot},{seed},{max_succ},{min_fail}"
            lines.append(new_line)
        else:
            # parse existing
            row = lines[found_idx].split(",")
            max_succ_str = row[4]
            min_fail_str = row[5]
            # update
            if success:
                # update max if bsz is bigger
                if max_succ_str == "":
                    row[4] = str(bsz)
                else:
                    prev = int(max_succ_str)
                    if bsz > prev:
                        row[4] = str(bsz)
            elif is_oom_error:
                # Only update min if this is an OOM error
                if min_fail_str == "":
                    row[5] = str(bsz)
                else:
                    prev = int(min_fail_str)
                    if bsz < prev:
                        row[5] = str(bsz)
            # If not success and not OOM, we don't update Min_Fail_BSZ
            lines[found_idx] = ",".join(row)
        
        write_best_fail_tracker(lines)

# ---------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    """
    Sets up a logger that writes to a file named:
      logs/{hostname}_{timestamp}.log
    Also logs to console, but we won't attempt to strip out tqdm lines here.
    
    Returns the logger object.
    """
    os.makedirs("logs", exist_ok=True)
    script_name = get_shell_script_name()
    gpu_label = Path(script_name).stem if script_name else "unknown"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"logs/{gpu_label}_{now_str}.log"
    
    log_format = "[%(asctime)s %(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(filename, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ---------------------------------------------------------------------
def get_shell_script_name() -> Optional[str]:
    """
    Attempt to figure out which shell script is calling us,
    by checking environment variables. 
    If not found, returns None.
    
    Some shells define BASH_SOURCE or $0 in different ways.
    We'll do a best-effort approach.
    """
    # In many setups, BASH_SOURCE[0] might hold something like: /path/to/h100.sh
    # Another approach: parse /proc/$PPID/cmdline
    script = os.environ.get("BASH_SOURCE")
    if script:
        return os.path.abspath(script)
    
    # Fallback: see if parent's command line references h100.sh, etc
    ppid = os.getppid()
    try:
        cmdline_path = f"/proc/{ppid}/cmdline"
        with open(cmdline_path, "r", encoding="utf-8") as f:
            content = f.read()
        # content is null-delimited
        # e.g. 'bash\x00h100.sh\x00'
        # We'll see if there's something with .sh
        parts = content.split('\x00')
        for p in parts:
            if p.endswith(".sh"):
                return os.path.abspath(p)
    except:
        pass
    
    return None

# ---------------------------------------------------------------------
class JobTracker:
    """
    A helper that:
      - Figures out which shell script is running,
      - Tracks best/fail BSZ,
      - Writes logs.
    """

    def __init__(self, logger: logging.Logger, args):
        self.logger = logger
        self.args = args

        # Extract the relevant fields
        self.model = args.model
        self.engine = args.engine
        self.tasks = args.tasks
        self.fewshot = args.num_fewshot
        self.batch_size = args.batch_size
        self.seed = args.seed

        self.hostname = os.uname()[1]  # e.g. 'h100'
        self.script_path = get_shell_script_name()
        self.gpu_label = Path(self.script_path).stem if self.script_path else "unknown"

        self.tasks_cat = determine_tasks_category(self.tasks)

        # For compatibility with old logic
        self.match_index = -1
        self.all_commands = []

    # -----------------------------------------------------------------
    def _parse_args_from_command_line(self, cmdline: str):
        """
        Unchanged helper (still used by internal logic),
        kept to minimise diff even though GPU status is gone.
        """
        parts = cmdline.split()
        md = None
        eg = None
        tasks_list = []
        fs = None
        bsz = None
        sd = None

        i = 0
        while i < len(parts):
            p = parts[i]
            if p == "--model":
                i += 1
                if i < len(parts):
                    md = parts[i].strip('"\'')
            elif p == "--engine":
                i += 1
                if i < len(parts):
                    eg = parts[i].strip('"\'')
            elif p == "--tasks":
                i += 1
                while i < len(parts) and not parts[i].startswith("--"):
                    tasks_list.append(parts[i].strip('"\''))
                    i += 1
                i -= 1
            elif p == "--num_fewshot":
                i += 1
                if i < len(parts):
                    fs = parts[i]
            elif p == "--batch_size":
                i += 1
                if i < len(parts):
                    bsz = parts[i]
            elif p == "--seed":
                i += 1
                if i < len(parts):
                    sd = parts[i]
            i += 1

        if not (md and eg and fs and bsz and sd is not None):
            return None

        tasks_str = " ".join(tasks_list) if tasks_list else ""
        return (md, eg, tasks_str, fs, bsz, sd)

    # -----------------------------------------------------------------
    def start_job(self):
        """
        Called at the start of main() after parsing arguments.
        - Initialises best_fail_tracker (no global/GPU tables).
        """
        init_best_fail_tracker()

    # -----------------------------------------------------------------
    def finish_job(self, success: bool, is_oom_error: bool = False):
        """
        Called at the end of main(), inside finally or except.
        - Updates best/fail tracker.
        - Only updates Min_Fail_BSZ if is_oom_error=True
        """
        update_best_fail_tracker(
            self.model,
            self.tasks_cat,
            self.fewshot,
            self.seed,
            self.batch_size,
            success,
            is_oom_error
        )