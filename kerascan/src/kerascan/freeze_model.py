"""Explicit local release step for an approved development/calibration model bundle."""
from __future__ import annotations
import argparse
from .model_bundle import ModelBundleError, load_model_bundle, save_model_bundle

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="Freeze an approved local KERASCAN model for evaluation.")
    parser.add_argument("--input-model",required=True);parser.add_argument("--output-model",required=True);parser.add_argument("--model-version",required=True);parser.add_argument("--confirm-approved-release",action="store_true")
    args=parser.parse_args(argv)
    if not args.confirm_approved_release: parser.error("Freezing requires explicit --confirm-approved-release after governance review.")
    try:
        bundle=load_model_bundle(args.input_model)
        if bundle.frozen: raise ModelBundleError("Input bundle is already frozen; do not rewrite an evaluation model.")
        try:
            from app.services.referral_engine import ReferralEngine
            bundle.protocol_version=ReferralEngine().get_protocol_version()
        except ImportError:
            pass
        bundle.frozen=True;bundle.model_version=args.model_version
        save_model_bundle(bundle,args.output_model)
        print("Approved local model frozen. Do not modify it before locked evaluation.")
        return 0
    except ModelBundleError as error: print(f"Model freeze stopped: {error}");return 2
if __name__=="__main__":raise SystemExit(main())
