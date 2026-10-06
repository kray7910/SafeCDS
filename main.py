import os
import json
import time
from src.reasoning.verifier import ClinicalVerifier
from src.agents.workflow import build_safecds_graph

ONTOLOGY_PATH = os.path.join("ontologies", "cardiometabolic_core.owl")


def main():
    print("Initializing SafeCDS Reasoning Engine...")
    verifier = ClinicalVerifier(ontology_path=ONTOLOGY_PATH)

    print("Building LangGraph Orchestration State Machine...")
    safecds_app = build_safecds_graph(verifier)

    # Test Patient: Patient presenting with CKD Stage 4/5 (where standard Type-2 diabetes first-line Metformin is unsafe)
    patient_input = {
        "patient_id": "Patient_Cardio_102",
        "conditions": ["CKD_Stage_4_5", "Type_2_Diabetes"],
        "retrieved_context": None,
        "proposed_medication": None,
        "reasoning": None,
        "retries": 0,
        "max_retries": 3,
        "violations": [],
        "status": "PENDING",
        "final_output": {}
    }

    print("\nExecuting SafeCDS Pipeline with Dynamic LLaMA-3.2 Reasoning...")
    start_t = time.perf_counter()
    result = safecds_app.invoke(patient_input)
    elapsed_ms = (time.perf_counter() - start_t) * 1000

    print("\n" + "=" * 45)
    print("             FINAL DECISION AUDIT            ")
    print("=" * 45)
    print(json.dumps(result["final_output"], indent=4))
    print(f"\nExecution Latency: {elapsed_ms:.2f} ms")
    print(f"Verifier Cache Stats: {verifier.cache_stats}")


if __name__ == "__main__":
    main()