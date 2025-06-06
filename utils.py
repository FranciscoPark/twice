import os
import json
from typing import Dict, Optional,List,Tuple
from datetime import datetime


def save_results(
    results: Dict[str, Dict[str, Tuple[float]]],
    model_name: str,
    tasks: List[str],
    output_dir: Optional[str] = "",
    seed: Optional[int] = 1234,
    shot: Optional[int] = 0
) -> None:
    """Save evaluation results to a JSON file."""
    if not output_dir:
        output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    safe_model_name = model_name.replace("/", "_")
    now = datetime.now()
    time =now.strftime("%Y-%m-%d %H:%M:%S")
    #print(time)  # e.g., 2025-05-29 15:24:00.123456
    if tasks[0]== 'mmlu':
        file_path = os.path.join(output_dir, f"{safe_model_name}-{shot}-{seed}-mmlu-{time}.json")
    else:
        file_path = os.path.join(output_dir, f"{safe_model_name}-{shot}-{seed}-{time}.json")

    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)