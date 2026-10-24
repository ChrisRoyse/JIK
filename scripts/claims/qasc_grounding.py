"""Source agreement is not world validity or a checked inference.

The QASC provider supplies two premises and a combined assertion. This adapter
records their exact provenance without promoting any of them to grounded memory.
It is a supervision/audit sidecar, never an input feature, sense, or learned judge.
Cost: three hashes per source row, once; constant-size lookup per rendered attempt.
"""
import hashlib

SOURCE_SHA256='b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1'

def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def source_records(source):
    return [dict(source='allenai/qasc',revision='a34ba204eb9a33b919c10cc08f4f1c8dae5ec070',
                 artifact='qasc-hf-training-v1/train.parquet',artifact_sha256=SOURCE_SHA256,row=i,
                 fields={name:text_sha(row[name]) for name in ('fact1','fact2','combinedfact')},
                 statement_kind='provider_assertion',world_validity='unadjudicated',
                 derivation_status='provider_supplied_unchecked',event_time=None,event_time_status='not_provided')
            for i,row in enumerate(source)]

def assessment(text,target,source_row):
    present=text is not None
    exact=present and text==target
    return dict(claim_sha256=text_sha(text) if present else None,source_record=source_row,
                source_agreement='exact_provider_assertion' if exact else 'unassessed' if present else 'no_complete_claim',
                source_agreement_is_world_validity=False,
                grounding_status='unadjudicated' if present else 'not_applicable_no_claim',
                memory_layer='episodic',promotion_eligible=False,
                promotion_blockers=['BASE_FACTS_UNADJUDICATED','DERIVATION_UNCHECKED'] if present else ['NO_COMPLETE_CLAIM'],
                truth_value='unknown',independent_grounding_receipts=[],checked_derivation=None,
                generated_claim_link='exact_provider_bytes' if exact else 'semantic_link_unchecked' if present else 'absent')
