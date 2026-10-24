"""Source-checkable tool episodes; recorded contents do not prove occurrence."""
from claims.codata_source import check
from claims.episodic_memory import digest


class ToolEvidence:
    def __init__(self, source, observations=None, conditions=None):
        self.source = source
        self.observations = observations
        self.conditions = conditions

    def condition_registry(self):
        check(self.conditions is not None or self.observations is not None, "TOOL_NO_EXECUTION_CHECKER")
        return self.conditions if self.conditions is not None else self.observations.condition_inputs()

    def episode(self, task, turn, evidence=()):
        ref = self.source.reference(task, turn)
        body = dict(claim_kind="source_statement", content=self.source.texts[task, turn],
                    source=ref, evidence=list(evidence), event_time=None,
                    who=dict(provider=self.source.by_task[task]["provider"], actor=None),
                    what=ref["role"], where=None,
                    when=dict(clock=ref["clock"], turn=turn, occurred_at=None),
                    why=None, why_status="unresolved", independent_occurrence=False)
        return dict(id=digest(body), **body)

    def validate_episode(self, episode):
        if episode.get("claim_kind") == "execution_precondition":
            self.condition_registry().validate_episode(episode)
            return
        if episode.get("claim_kind") == "event_occurrence":
            check(self.observations is not None, "TOOL_NO_EXECUTION_CHECKER")
            self.observations.validate_episode(episode)
            return
        ref = episode.get("source", {})
        check(episode == self.episode(ref.get("task"), ref.get("turn"), episode.get("evidence", ())),
              "TOOL_EPISODE")
        check(len(set(episode["evidence"])) == len(episode["evidence"]), "TOOL_EVIDENCE_DUPLICATE")

    def assess(self, episode):
        self.validate_episode(episode)
        if episode["claim_kind"] == "execution_precondition":
            return self.condition_registry().assess(episode)
        if episode["claim_kind"] == "event_occurrence":
            return self.observations.assess(episode)
        body = dict(checker="pinned-tool-episode-v1", episode=episode["id"],
                    claim_kind="source_statement", scope="pinned_artifact_contents_only",
                    source=episode["source"], world_validity="unassessed")
        return dict(id=digest(body), **body)
