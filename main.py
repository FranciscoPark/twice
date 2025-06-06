#check if dataset is availabe in db yes -> run with TIE(twice inference engine), no -> mcp server


import os
import sys
import argparse
from typing import List
import traceback
import torch

# 1) Import your new job_logger
from job_logger import setup_logger, JobTracker

from cli_evaluator import evaluate
from loader.task_config import ALL_TASKS

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified evaluation script for language models")

    parser.add_argument(
        "--model", 
        type=str,
        default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        default=["all"],
        help="Task(s) to evaluate. Use 'all' for all tasks."
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0,
        help="Number of examples to use for few-shot learning"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1234,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="the number of data to evaluate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="the batch size"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    return parser

def resolve_tasks(args: argparse.Namespace) -> List[str]:
    if 'all' in args.tasks:
        return ALL_TASKS
    else:
        return args.tasks

def main():
    # 2) Set up logger at the very beginning
    logger = setup_logger()

    parser = setup_parser()
    args, engine_args = parser.parse_known_args()

    # 3) Print/Log the arguments at the beginning of the log
    logger.info(f"Parsed arguments: {vars(args)}")
    if engine_args:
        logger.info(f"Extra engine args: {engine_args}")

    # 4) Initialize the job tracker
    job_tracker = JobTracker(logger, args)
    job_tracker.start_job()

    tasks = resolve_tasks(args)

    try:
        # 5) Perform the evaluation
        if args.engine == 'hf':
            evaluate(args, tasks)
        else:
            # For your vllm usage
            evaluate(args, tasks, engine_args)

        # If we reach here, success
        job_tracker.finish_job(success=True)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Check if it's an OOM error
        is_oom = False
        if isinstance(e, torch.cuda.OutOfMemoryError):
            is_oom = True
        elif isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
            is_oom = True
        
        # Log and print the error
        error_traceback = traceback.format_exc()
        logger.error("Exception during evaluate():")
        logger.error(error_traceback)
        
        # Also print to terminal (stderr) for immediate visibility
        print("\n" + "="*80, file=sys.stderr)
        if is_oom:
            print("ERROR: Out of Memory (OOM) error occurred during evaluation!", file=sys.stderr)
        else:
            print("ERROR: RuntimeError occurred during evaluation!", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(error_traceback, file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        # Only update Min_Fail_BSZ for OOM errors
        job_tracker.finish_job(success=False, is_oom_error=is_oom)
        sys.exit(1)
        
    except KeyboardInterrupt:
        # Handle Ctrl+C separately - don't update Min_Fail_BSZ
        logger.error("Evaluation interrupted by user (Ctrl+C)")
        print("\n" + "="*80, file=sys.stderr)
        print("ERROR: Evaluation interrupted by user (Ctrl+C)", file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        # Don't update tracker for manual interruption
        job_tracker.finish_job(success=False, is_oom_error=False)
        sys.exit(1)
        
    except Exception as e:
        # 6) Other errors - don't update Min_Fail_BSZ
        error_traceback = traceback.format_exc()
        logger.error("Exception during evaluate():")
        logger.error(error_traceback)
        
        # Also print to terminal (stderr) for immediate visibility
        print("\n" + "="*80, file=sys.stderr)
        print("ERROR: Exception occurred during evaluation!", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(error_traceback, file=sys.stderr)
        print("="*80, file=sys.stderr)
        
        # Don't update Min_Fail_BSZ for other errors
        job_tracker.finish_job(success=False, is_oom_error=False)
        sys.exit(1)

if __name__ == "__main__":
    main()