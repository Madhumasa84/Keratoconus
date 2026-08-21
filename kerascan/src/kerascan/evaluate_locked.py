"""Explicitly guarded CLI for one-way locked-test evaluation."""
from __future__ import annotations
import argparse
from .evaluate import run_evaluation
from .manifest import ManifestValidationError
from .model_bundle import ModelBundleError

WARNING="Do not modify or retune the model based on locked-test results. Any subsequent model must be evaluated on a new untouched test set."

def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description="Explicit locked-test evaluation; local only.")
    parser.add_argument("--manifest",required=True);parser.add_argument("--model",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--confirm-locked-evaluation",action="store_true");parser.add_argument("--development-manifest");parser.add_argument("--calibration-manifest");parser.add_argument("--bootstrap-iterations",type=int,default=1000)
    args=parser.parse_args(argv)
    if not args.confirm_locked_evaluation:
        parser.error("Locked data are evaluation-only. Re-run with --confirm-locked-evaluation.")
    print("WARNING: "+WARNING)
    try:
        result=run_evaluation(args.manifest,args.model,args.output,locked=True,development_manifest=args.development_manifest,calibration_manifest=args.calibration_manifest,bootstrap_iterations=args.bootstrap_iterations)
        print(f"Locked evaluation completed locally. Target achieved: {result['metrics'].get('accuracy') is not None and result['metrics']['accuracy'] > .95}. {WARNING}")
        return 0
    except (ManifestValidationError,ModelBundleError,ValueError) as error:
        print(f"Locked evaluation stopped: {error}");return 2

if __name__=="__main__": raise SystemExit(main())
