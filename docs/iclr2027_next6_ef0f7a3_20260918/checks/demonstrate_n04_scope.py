#!/usr/bin/env python3
"""Reproduce the Python closure shadowing pattern found in N04.
This does not import or rerun the project's model; it tests language semantics only.
"""
import json

def reproduce():
    sd = [0.2, 0.5]
    stats = {"sd": list(sd)}
    def training_divisor():
        return sd
    rows = []
    for sd in [11, 23, 47]:
        rows.append({"seed": sd, "training_divisor": training_divisor(),
                     "evaluation_divisor": stats["sd"]})
    return rows

def corrected():
    feature_std = [0.2, 0.5]
    def training_divisor():
        return feature_std
    rows = []
    for seed_id in [11, 23, 47]:
        rows.append({"seed": seed_id, "training_divisor": training_divisor(),
                     "evaluation_divisor": feature_std})
    return rows

if __name__ == "__main__":
    bad, good = reproduce(), corrected()
    assert [r["training_divisor"] for r in bad] == [11, 23, 47]
    assert all(r["training_divisor"] == r["evaluation_divisor"] for r in good)
    print(json.dumps({"scope_reproduction": bad, "renamed_variables": good}, indent=2))
    print("PASS: closure shadowing reproduced; renamed variables resolve the scope issue.")
