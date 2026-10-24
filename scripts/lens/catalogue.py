"""The Stage 3 candidate catalogue (issue #12, spec §19.2 Table 3): every candidate lens with the inputs that define its
contract. Learned lenses name a Hugging Face repository and a pinned revision (read back at registration from the snapshot;
the provider's training corpus is not disclosed, so `corpus_hash` is the repository commit). Algorithmic lenses are the
encoders of encoders.py, identified by that file's digest. Nothing here is admitted; admission is measured (assay.py)."""
from __future__ import annotations

# kind: dense (pooled vector), sparse (vocabulary weights), multi (token vectors), rater (classifier logits), nll (mean token NLL), static (static token table)
# pool: mean | cls | last ; prefix: prepended to every input ; max_len: tokens per window
LEARNED: list[dict] = [
    {"name": "e5-multilingual-base", "repo": "intfloat/multilingual-e5-base", "kind": "dense", "pool": "mean", "prefix": "query: ", "max_len": 512, "group": "general"},
    {"name": "e5-multilingual-small", "repo": "intfloat/multilingual-e5-small", "kind": "dense", "pool": "mean", "prefix": "query: ", "max_len": 512, "group": "general"},
    {"name": "e5-base-v2", "repo": "intfloat/e5-base-v2", "kind": "dense", "pool": "mean", "prefix": "query: ", "max_len": 512, "group": "general"},
    {"name": "bge-base-en-v1.5", "repo": "BAAI/bge-base-en-v1.5", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 512, "group": "general"},
    {"name": "bge-m3-dense", "repo": "BAAI/bge-m3", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 1024, "group": "general"},
    {"name": "bge-m3-sparse", "repo": "BAAI/bge-m3", "kind": "sparse", "head": "sparse_linear.pt", "prefix": "", "max_len": 1024, "group": "lexical"},
    {"name": "bge-m3-colbert", "repo": "BAAI/bge-m3", "kind": "multi", "head": "colbert_linear.pt", "prefix": "", "max_len": 1024, "group": "late-interaction"},
    {"name": "arctic-embed-l-v2", "repo": "Snowflake/snowflake-arctic-embed-l-v2.0", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 512, "group": "general"},
    {"name": "granite-278m-multilingual", "repo": "ibm-granite/granite-embedding-278m-multilingual", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 512, "group": "general"},
    {"name": "granite-english-r2", "repo": "ibm-granite/granite-embedding-english-r2", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 1024, "group": "general"},
    {"name": "mxbai-embed-large-v1", "repo": "mixedbread-ai/mxbai-embed-large-v1", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 512, "group": "general"},
    {"name": "all-minilm-l6-v2", "repo": "sentence-transformers/all-MiniLM-L6-v2", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 256, "group": "general"},
    {"name": "all-mpnet-base-v2", "repo": "sentence-transformers/all-mpnet-base-v2", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 384, "group": "general"},
    {"name": "paraphrase-multilingual-minilm", "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 128, "group": "general"},
    {"name": "nomic-modernbert-embed", "repo": "nomic-ai/modernbert-embed-base", "kind": "dense", "pool": "mean", "prefix": "search_document: ", "max_len": 1024, "group": "general"},
    {"name": "gte-modernbert-base", "repo": "Alibaba-NLP/gte-modernbert-base", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 1024, "group": "general"},
    {"name": "gte-base", "repo": "thenlper/gte-base", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "general"},
    {"name": "qwen3-embedding-0.6b", "repo": "Qwen/Qwen3-Embedding-0.6B", "kind": "dense", "pool": "last", "prefix": "", "max_len": 1024, "group": "general"},
    {"name": "specter2-base", "repo": "allenai/specter2_base", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 512, "group": "domain-science"},
    {"name": "allenai-specter", "repo": "sentence-transformers/allenai-specter", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "domain-science"},
    {"name": "scibert-mean", "repo": "allenai/scibert_scivocab_uncased", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "domain-science"},
    {"name": "medembed-base", "repo": "abhinand/MedEmbed-base-v0.1", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 512, "group": "domain-biomedical"},
    {"name": "pubmedbert-embeddings", "repo": "NeuML/pubmedbert-base-embeddings", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "domain-biomedical"},
    {"name": "legal-bert-mean", "repo": "nlpaueb/legal-bert-base-uncased", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "domain-legal"},
    {"name": "finbert-mean", "repo": "ProsusAI/finbert", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "domain-finance"},
    {"name": "bge-financial-matryoshka", "repo": "philschmid/bge-base-financial-matryoshka", "kind": "dense", "pool": "cls", "prefix": "", "max_len": 512, "group": "domain-finance"},
    {"name": "unixcoder-mean", "repo": "microsoft/unixcoder-base", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "domain-code"},
    {"name": "codebert-mean", "repo": "microsoft/codebert-base", "kind": "dense", "pool": "mean", "prefix": "", "max_len": 512, "group": "domain-code"},
    {"name": "potion-base-8m", "repo": "minishlab/potion-base-8M", "kind": "static", "prefix": "", "max_len": 4096, "group": "general"},
    {"name": "splade-pp-en-v1", "repo": "prithivida/Splade_PP_en_v1", "kind": "sparse", "prefix": "", "max_len": 512, "group": "lexical"},
    {"name": "opensearch-sparse-doc-v2", "repo": "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill", "kind": "sparse", "prefix": "", "max_len": 512, "group": "lexical"},
    {"name": "colbertv2", "repo": "colbert-ir/colbertv2.0", "kind": "multi", "head": "linear.weight", "prefix": "", "max_len": 512, "group": "late-interaction"},
    {"name": "answerai-colbert-small", "repo": "answerdotai/answerai-colbert-small-v1", "kind": "multi", "head": "linear.weight", "prefix": "", "max_len": 512, "group": "late-interaction"},
    {"name": "gte-moderncolbert", "repo": "lightonai/GTE-ModernColBERT-v1", "kind": "multi", "head": "1_Dense/model.safetensors", "prefix": "", "max_len": 1024, "group": "late-interaction"},
    {"name": "fineweb-edu-classifier", "repo": "HuggingFaceTB/fineweb-edu-classifier", "kind": "rater", "prefix": "", "max_len": 512, "group": "quality-rater"},
    {"name": "gpt2-nll", "repo": "openai-community/gpt2", "kind": "nll", "prefix": "", "max_len": 1024, "group": "quality-rater"},
    {"name": "twitter-roberta-sentiment", "repo": "cardiffnlp/twitter-roberta-base-sentiment-latest", "kind": "rater", "prefix": "", "max_len": 512, "group": "affect"},
    {"name": "roberta-go-emotions", "repo": "SamLowe/roberta-base-go_emotions", "kind": "rater", "prefix": "", "max_len": 512, "group": "affect"},
    {"name": "emotion-distilroberta", "repo": "j-hartmann/emotion-english-distilroberta-base", "kind": "rater", "prefix": "", "max_len": 512, "group": "affect"},
    {"name": "formality-ranker", "repo": "s-nlp/roberta-base-formality-ranker", "kind": "rater", "prefix": "", "max_len": 512, "group": "register"},
    {"name": "xlmr-language-detection", "repo": "papluca/xlm-roberta-base-language-detection", "kind": "rater", "prefix": "", "max_len": 512, "group": "register"},
    {"name": "chatgpt-detector-roberta", "repo": "Hello-SimpleAI/chatgpt-detector-roberta", "kind": "rater", "prefix": "", "max_len": 512, "group": "register"},
    {"name": "roberta-cola", "repo": "textattack/roberta-base-CoLA", "kind": "rater", "prefix": "", "max_len": 512, "group": "register"},
    {"name": "toxic-bert", "repo": "unitary/toxic-bert", "kind": "rater", "prefix": "", "max_len": 512, "group": "affect"},
]

# Algorithmic lenses (encoders.py). `inputs`: fields (explicit record fields) | text | code. `dims` is a nominal upper bound.
ALGORITHMIC: list[dict] = [
    {"name": "scalar-raw", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "scalar-minmax", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "scalar-z", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "scalar-log1p", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "scalar-rank", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "scalar-robust", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "scalar-winsor", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "cyclic-calendar", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "temporal-clock"},
    {"name": "clock-knowledge-age", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "temporal-clock"},
    {"name": "geo-latlong", "kind": "encoder", "inputs": "fields", "sim": "neg_cheb", "group": "structured"},
    {"name": "onehot-lowcard", "kind": "encoder", "inputs": "fields", "sim": "cosine", "group": "structured"},
    {"name": "hash-highcard", "kind": "encoder", "inputs": "fields", "sim": "cosine", "group": "structured"},
    {"name": "record-vector", "kind": "encoder", "inputs": "fields", "sim": "cosine", "group": "structured", "baseline": True},
    {"name": "provenance-source-class", "kind": "encoder", "inputs": "fields", "sim": "cosine", "group": "provenance"},
    {"name": "length-signature", "kind": "encoder", "inputs": "text", "sim": "neg_cheb", "group": "stylometry"},
    {"name": "text-stylometry", "kind": "encoder", "inputs": "text", "sim": "cosine", "group": "stylometry"},
    {"name": "text-hash-tfidf", "kind": "encoder", "inputs": "text", "sim": "cosine", "group": "lexical", "sparse": True},
    {"name": "text-char-trigram", "kind": "encoder", "inputs": "text", "sim": "cosine", "group": "lexical", "sparse": True},
    {"name": "code-ast-histogram", "kind": "encoder", "inputs": "code", "sim": "cosine", "group": "code-structure"},
    {"name": "code-structure-signature", "kind": "encoder", "inputs": "code", "sim": "neg_cheb", "group": "code-structure"},
]

# Context lenses (encoders_ctx.py): deterministic encoders over the source-backed context beside the atom (prior GDELT days,
# the Wikidata5m triple store, the Stage 2 citation out-edges, the module's static surface). `source` names the atoms they read.
CONTEXT: list[dict] = [
    {"name": "gdelt-pair-history", "kind": "encoder", "inputs": "context", "sim": "neg_cheb", "group": "relational-aggregate", "source": "gdelt-events-90d"},
    {"name": "gdelt-actor-activity", "kind": "encoder", "inputs": "context", "sim": "neg_cheb", "group": "relational-aggregate", "source": "gdelt-events-90d"},
    {"name": "gdelt-event-recurrence", "kind": "encoder", "inputs": "context", "sim": "neg_cheb", "group": "relational-aggregate", "source": "gdelt-events-90d"},
    {"name": "gdelt-pair-tone-mix", "kind": "encoder", "inputs": "context", "sim": "neg_cheb", "group": "relational-aggregate", "source": "gdelt-events-90d"},
    {"name": "kg-relation-signature", "kind": "encoder", "inputs": "context", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": ["a2-wikidata-p31", "a2-wikidata-p31-rare"]},
    {"name": "kg-neighbour-classes", "kind": "encoder", "inputs": "context", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": ["a2-wikidata-p31", "a2-wikidata-p31-rare"]},
    {"name": "cite-out-signature", "kind": "encoder", "inputs": "context", "sim": "neg_cheb", "group": "graph-structural", "source": "cuyahoga-eighth-district"},
    {"name": "code-static-sensors", "kind": "encoder", "inputs": "context", "sim": "neg_cheb", "group": "code-structure", "source": "opencodeinstruct",
     "only_families": ["a1-oci-tests", "a1-oci-outcome3", "a1-oci-ran-tests"]},  # census 2026-10-17: on prose it reads a line count, not static structure
]

# Label-free graph lenses for the other A2 statement questions (encoders_kg.py; readback finding F5): the first pair excludes
# the entity's own P31 and is restricted to the P31 questions; this pair excludes the asked property and serves P17/P106.
CONTEXT += [
    {"name": "kg-relation-signature-q", "kind": "encoder", "inputs": "context", "module": "encoders_kg", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": ["a2-wikidata-p17", "a2-wikidata-p106"]},
    {"name": "kg-neighbour-classes-q", "kind": "encoder", "inputs": "context", "module": "encoders_kg", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": ["a2-wikidata-p17", "a2-wikidata-p106"]},
]

# Second round (encoders_ctx2.py, after admission run 6): each reads an input no registered lens reads (the A1 task prompt, the
# A4 chapter neighbours in the same edition, the GDELT source domain's and action country's prior days, the two-hop citation
# authority and judge roster of an A6 opinion, the asked property's values over an A2 entity's neighbours).
CONTEXT += [
    {"name": "code-prompt-signature", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "cosine", "group": "task-prompt", "source": "opencodeinstruct", "sparse": True},
    {"name": "code-prompt-sensors", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "neg_cheb", "group": "task-prompt", "source": "opencodeinstruct"},
    {"name": "scripture-chapter-context", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "cosine", "group": "relational-aggregate", "source": "crosswire-sword-pd", "sparse": True,
     "retired": "decision D-2 (#12, readback G1): reads the label's siblings (neighbours share the unit the label is a function of)"},
    {"name": "scripture-chapter-shape", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "neg_cheb", "group": "relational-aggregate", "source": "crosswire-sword-pd",
     "retired": "decision D-2 (#12, readback G1): reads the label's siblings (neighbours share the unit the label is a function of)"},
    {"name": "gdelt-source-history", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "neg_cheb", "group": "relational-aggregate", "source": "gdelt-events-90d"},
    {"name": "gdelt-geo-history", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "neg_cheb", "group": "relational-aggregate", "source": "gdelt-events-90d"},
    {"name": "cite-authority", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "neg_cheb", "group": "graph-structural", "source": "cuyahoga-eighth-district"},
    {"name": "cite-roster", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "cosine", "group": "who-only-roster", "source": "cuyahoga-eighth-district", "sparse": True},
    {"name": "kg-neighbour-values-q", "kind": "encoder", "inputs": "context", "module": "encoders_ctx2", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": ["a2-wikidata-p17", "a2-wikidata-p106"],
     "retired": "decision D-2 (#12, readback G1): reads the label's siblings (neighbours share the unit the label is a function of)"},
]

# Third round (encoders_ctx3.py, Johari 2 solution 2): inputs no registered lens reads (the A2 abstract outside the atom span, the
# A1 unit tests and the solution's own execution trace, the A6 disposition and party structure) and the graph pair for the six
# new A2 questions (the asked property excluded in either direction, as in encoders_kg.py / kg-neighbour-values-q).
A2_NEW = ["a2-wikidata-p27", "a2-wikidata-p19", "a2-wikidata-p69", "a2-wikidata-p641", "a2-wikidata-p136", "a2-wikidata-p495"]
A2_ALL = ["a2-wikidata-p31", "a2-wikidata-p31-rare", "a2-wikidata-p17", "a2-wikidata-p106", *A2_NEW]
CONTEXT += [
    {"name": "a2-abstract-rest", "kind": "encoder", "inputs": "context", "module": "encoders_ctx3", "sim": "cosine", "group": "lexical-context", "source": "wikidata5m", "sparse": True, "only_families": A2_ALL,
     "retired": "readback J1 (run 10): the context names the question's own class (article body 83 % of P54 rows; probe over all A2 questions 10-96 %), re-reading the oracle's source; replaced by its -masked lens"},
    {"name": "code-tests-signature", "kind": "encoder", "inputs": "context", "module": "encoders_ctx3", "sim": "cosine", "group": "task-tests", "source": "opencodeinstruct", "sparse": True},
    {"name": "code-exec-trace", "kind": "encoder", "inputs": "context", "module": "encoders_ctx3", "sim": "neg_cheb", "group": "code-structure", "source": "opencodeinstruct",
     "retired": "readback H1 (runs 8-9): its ran feature is the A1 oracle load-failure clause; replaced by code-run-trace on a1-oci-ran-tests"},
    {"name": "opinion-disposition-sensors", "kind": "encoder", "inputs": "context", "module": "encoders_ctx3", "sim": "neg_cheb", "group": "typed-structure", "source": "cuyahoga-eighth-district"},
    {"name": "kg-relation-signature-q2", "kind": "encoder", "inputs": "context", "module": "encoders_ctx3", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": A2_NEW},
    {"name": "kg-neighbour-classes-q2", "kind": "encoder", "inputs": "context", "module": "encoders_ctx3", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": A2_NEW},
    {"name": "kg-neighbour-values-q2", "kind": "encoder", "inputs": "context", "module": "encoders_ctx3", "sim": "cosine", "group": "graph-structural", "source": "wikidata5m", "only_families": A2_NEW,
     "retired": "decision D-2 (#12, readback G1): reads the label's siblings (neighbours share the unit the label is a function of)"},
]

# Cross-source perspectives (encoders_ctx4.py; operator doctrine 2026-10-16): the same referent sensed through another source.
A2_ALL += ["a2-wikidata-p54", "a2-wikidata-p166", "a2-wikidata-p463", "a2-wikidata-p108", "a2-wikidata-p1412", "a2-wikidata-p131"]
CONTEXT += [
    {"name": "wiki-article-body", "kind": "encoder", "inputs": "context", "module": "encoders_ctx4", "sim": "cosine", "group": "cross-source", "source": "wikidata5m", "sparse": True, "only_families": A2_ALL,
     "retired": "readback J1 (run 10): the context names the question's own class (article body 83 % of P54 rows; probe over all A2 questions 10-96 %), re-reading the oracle's source; replaced by its -masked lens"},
    {"name": "wiki-mentions", "kind": "encoder", "inputs": "context", "module": "encoders_ctx4", "sim": "cosine", "group": "cross-source", "source": "wikidata5m", "sparse": True, "only_families": A2_ALL,
     "retired": "readback J1 (run 10): the context names the question's own class (article body 83 % of P54 rows; probe over all A2 questions 10-96 %), re-reading the oracle's source; replaced by its -masked lens"},
    {"name": "scripture-other-editions", "kind": "encoder", "inputs": "context", "module": "encoders_ctx4", "sim": "cosine", "group": "cross-source", "source": "crosswire-sword-pd", "sparse": True, "only_families": ["a4-scripture-genre", "a4-scripture-genre-keyfree"]},
]

CONTEXT += [
    {"name": "code-run-trace", "kind": "encoder", "inputs": "context", "module": "encoders_ctx5", "sim": "neg_cheb", "group": "code-structure", "source": "opencodeinstruct", "only_families": ["a1-oci-ran-tests"]},
]

# The A2 contexts with the answer masked (encoders_ctx6.py; readback J1 of run 10): every alias of every class the question declares
# is deleted from the text before it is read, so the lens senses what the context says besides the answer.
CONTEXT += [
    {"name": "wiki-article-body-masked", "kind": "encoder", "inputs": "context", "module": "encoders_ctx6", "sim": "cosine", "group": "cross-source", "source": "wikidata5m", "sparse": True, "only_families": A2_ALL},
    {"name": "wiki-mentions-masked", "kind": "encoder", "inputs": "context", "module": "encoders_ctx6", "sim": "cosine", "group": "cross-source", "source": "wikidata5m", "sparse": True, "only_families": A2_ALL},
    {"name": "a2-abstract-rest-masked", "kind": "encoder", "inputs": "context", "module": "encoders_ctx6", "sim": "cosine", "group": "lexical-context", "source": "wikidata5m", "sparse": True, "only_families": A2_ALL},
]

# Round 4 (encoders_ctx7.py; agent decisions D-2/D-7 of 2026-10-16): cross-references into other books (A4) and the entity's own names (A2).
CONTEXT += [
    {"name": "scripture-crossrefs", "kind": "encoder", "inputs": "context", "module": "encoders_ctx7", "sim": "cosine", "group": "cross-reference", "source": "crosswire-sword-pd", "sparse": True,
     "only_families": ["a4-scripture-genre-keyfree"]},
    {"name": "kg-entity-names", "kind": "encoder", "inputs": "context", "module": "encoders_ctx7", "sim": "cosine", "group": "entity-names", "source": "wikidata5m", "sparse": True, "only_families": A2_ALL},
]

CATALOGUE = LEARNED + ALGORITHMIC + CONTEXT
NAMES = [c["name"] for c in CATALOGUE]

# Perspectives: the declared input channel a lens reads, a descriptive label only. D4 counts admitted lenses: each lens
# that passes D1/D2 is its own sense, so two text embedders are two senses when both are admitted (operator 2026-10-18).
PERSPECTIVE_BY_NAME = {
    "code-ast-histogram": "static-structure", "code-structure-signature": "static-structure", "code-static-sensors": "static-structure",
    "code-exec-trace": "execution-trace", "code-run-trace": "execution-trace", "code-prompt-signature": "task-prompt", "code-prompt-sensors": "task-prompt", "code-tests-signature": "task-tests",
    "provenance-source-class": "provenance", "cite-out-signature": "citation-graph", "cite-authority": "citation-graph",
    "scripture-chapter-context": "document-context", "scripture-chapter-shape": "document-context", "a2-abstract-rest": "document-context",
    "wiki-article-body": "cross-source-article", "wiki-mentions": "cross-source-mentions",
    "scripture-crossrefs": "citation-graph", "kg-entity-names": "explicit-fields",
    "wiki-article-body-masked": "cross-source-article", "wiki-mentions-masked": "cross-source-mentions", "a2-abstract-rest-masked": "document-context", "scripture-other-editions": "cross-source-editions",
}
PERSPECTIVE_BY_GROUP = {"relational-aggregate": "history-clock", "graph-structural": "graph-neighbourhood", "temporal-clock": "explicit-fields", "structured": "explicit-fields"}


def perspective_of(spec: dict) -> str:
    if spec["name"] in PERSPECTIVE_BY_NAME:
        return PERSPECTIVE_BY_NAME[spec["name"]]
    if spec.get("group") in PERSPECTIVE_BY_GROUP:
        return PERSPECTIVE_BY_GROUP[spec["group"]]
    return "surface-text"  # every learned embedder, lexical/stylometric encoder and surface-regex sensor reads the atom's own surface


for _c in CATALOGUE:
    _c["perspective"] = perspective_of(_c)
    _c["role"] = "legacy_source_adapter" if _c.get("inputs") == "context" else "legacy_instrument"
# Source adapters do not earn another sense merely by supplying another view.
from universal import specs as universal_specs
CATALOGUE += universal_specs()
from learned_ucr import specs as learned_universal_specs
CATALOGUE += learned_universal_specs(LEARNED)
from audio_encoder import SPEC as AUDIO_SPEC
CATALOGUE.append(AUDIO_SPEC)
from image_ucr import SPEC as IMAGE_SPEC
CATALOGUE.append(IMAGE_SPEC)
from vision_encoder import SPEC as VISION_SPEC
CATALOGUE.append(VISION_SPEC)
NAMES = [c["name"] for c in CATALOGUE]
PERSPECTIVES = sorted({c["perspective"] for c in CATALOGUE})
assert len(NAMES) == len(set(NAMES)), "duplicate lens name in the catalogue"


def by_name(name: str) -> dict:
    for c in CATALOGUE:
        if c["name"] == name:
            return c
    raise KeyError(name)
