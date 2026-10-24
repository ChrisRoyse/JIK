"""Correctable speech-event evidence in the existing episodic Memory/EventJournal."""
import json
from pathlib import Path

from claims.episodic_memory import Memory, canonical, digest
from claims.event_journal import EventJournal
from senses.perception_corpus import sha
from senses.timed_speech import FLAGS, NITE, need, pair, rational

SCOPE = "source_published_speech_and_recording_views"


def episode(kind, content, source, event_time, evidence=()):
    body = dict(claim_kind=kind, content=content, source=source,
                event_time=event_time, evidence=list(evidence))
    return dict(id=digest(body), **body)


class SpeechEvidence:
    def __init__(self, source, views):
        self.source = source
        self.views = {v.manifest["event"]["segment"][NITE+"id"]: v for v in views}
        need(len(self.views) == len(views) and self.views, "VIEW_ROSTER")
        self.pin = digest(dict(source=source.pin,
            views={k: v.pin for k, v in self.views.items()}, code=sha(Path(__file__))))

    def parents(self, key):
        need(key in self.views, "UNKNOWN_SEGMENT")
        view = self.views[key]
        ev = view.manifest["event"]
        source = dict(bundle=self.source.pin, view=view.pin, segment=key,
                      clock_id=ev["clock_id"], split_group=ev["split_group"])
        time = dict(support=ev["support"], available_after=ev["available_after"])
        parents = []
        for medium in ("image", "audio", "text"):
            content = dict(what="source_published_transcription" if medium == "text"
                           else "decoded_recording_view",
                medium=medium, who=ev["speaker"], where=None, why=None,
                view=view.pin, native=ev[medium], flags=FLAGS)
            # Transcript remains a dependent account of audio, never a second witness.
            deps = [parents[1]["id"]] if medium == "text" else []
            parents.append(episode("source_statement" if medium == "text" else "measurement",
                                   content, source, time, deps))
        return tuple(parents)

    def moment(self, key, at):
        need(key in self.views, "UNKNOWN_SEGMENT")
        ev = self.views[key].manifest["event"]
        t = rational(at)
        need(rational(ev["support"][0]) <= t < rational(ev["support"][1]), "MOMENT_RANGE")
        image, audio, text = ev["image"], ev["audio"], ev["text"]
        tb = rational(image["time_base"])
        frames = [f for f in image["frames"] if f["pts"]*tb <= t
                  < (f["pts"]+f["duration"])*tb]
        words = [w for w in text["words"] if rational(w["span"][0]) <= t
                 < rational(w["span"][1])]
        need(len(frames) == 1 and words, "MISSING_ACTIVE_FRAME_OR_LANGUAGE")
        sample = (t*audio["rate"]).__floor__()
        need(audio["start_sample"] <= sample < audio["stop_sample"], "MISSING_ACTIVE_AUDIO")
        parents = self.parents(key)
        return episode("derived_relation", dict(what="source_attested_speech_event_constellation",
            who=ev["speaker"], where=None, why=None, at=at,
            image=dict(frame=frames[0], time_base=image["time_base"]),
            audio=dict(sample=sample, sample_time=pair(rational([sample, audio["rate"]]))),
            language=[w["attributes"][NITE+"id"] for w in words],
            context_support=ev["support"], timing=ev["timing"], flags=FLAGS),
            parents[0]["source"],
            dict(at=at, available_after=ev["available_after"]), [p["id"] for p in parents])

    def unresolved(self, moment):
        return episode("interpretation", dict(what="why_unresolved", explanation=None),
                       moment["source"], moment["event_time"], [moment["id"]])

    def validate_episode(self, ep):
        need(type(ep) == dict and set(ep) ==
             {"id", "claim_kind", "content", "source", "event_time", "evidence"}, "EPISODE_SCHEMA")
        key = ep["source"]["segment"]
        if ep["claim_kind"] in ("measurement", "source_statement"):
            candidates = self.parents(key)
            need(ep in candidates, "EPISODE_BINDING")
        elif ep["claim_kind"] == "derived_relation":
            need(ep == self.moment(key, ep["content"]["at"]), "EPISODE_BINDING")
        elif ep["claim_kind"] == "interpretation":
            need(ep == self.unresolved(self.moment(key, ep["event_time"]["at"])), "EPISODE_BINDING")
        else:
            need(False, "UNSUPPORTED_CLAIM")

    def assess(self, ep):
        self.validate_episode(ep)
        need(ep["claim_kind"] != "interpretation", "WHY_UNRESOLVED")
        body = dict(checker="source-published-timed-speech-v1", evidence=self.pin,
                    episode=ep["id"], scope=SCOPE, world_validity="unassessed",
                    source_timing="provider_forced_alignment", **FLAGS)
        return dict(id=digest(body), **body)


class SpeechMemory:
    def __init__(self, path, evidence, *, create, retained=None):
        need(create or (type(retained) == dict and set(retained) ==
                        {"head", "contract", "memory_head"}), "RETAINED_HEAD")
        self.evidence, self.memory = evidence, Memory(evidence)
        contract = dict(schema="timed-speech-memory-v1", evidence=evidence.pin, flags=FLAGS,
            code={name: sha(Path(__file__).with_name(name))
                  for name in ("timed_speech_memory.py", "episodic_memory.py", "event_journal.py")},
            adapter=sha(Path(__file__).parents[1]/"senses"/"timed_speech.py"))
        self.journal = EventJournal(path, contract, create=create, payload_bound=0)
        self.broken = False
        if not create:
            try:
                self.journal.restore(self._replay, head=retained["head"], contract=retained["contract"])
                need(self.memory.head == retained["memory_head"], "MEMORY_HEAD")
            except BaseException:
                if not self.journal.broken:
                    self.journal.close()
                self.broken = True
                raise

    def retained(self):
        need(not self.broken, "RELOAD")
        self.journal.fresh()
        return dict(head=self.journal.head, contract=self.journal.pin, memory_head=self.memory.head)

    def _checked(self, key):
        need(key in self.memory.state, "UNKNOWN_EPISODE")
        state = self.memory.state[key]
        need(state["active"] and state["grounded_scope"] == SCOPE
             and state["receipt"] == self.evidence.assess(self.memory.episodes[key]),
             "SUPPORT_WITHDRAWN_OR_UNCHECKED")

    def _dependencies(self, command):
        if command.get("op") in ("check_source", "promote"):
            key = command.get("episode")
            need(key in self.memory.episodes, "UNKNOWN_EPISODE")
            for parent in self.memory.episodes[key]["evidence"]:
                self._checked(parent)

    def _replay(self, body):
        need(type(body) == dict and set(body) == {"command", "record"}, "JOURNAL_EVENT")
        self._dependencies(body["command"])
        need(self.memory.apply(body["command"]) == body["record"], "MEMORY_REPLAY")

    def apply(self, command, *, expected_head):
        need(self.retained()["head"] == expected_head, "WRITER_CONFLICT")
        self._dependencies(command)
        record = self.memory.apply(command)
        try:
            self.journal.append(dict(command=json.loads(canonical(command)), record=record))
        except BaseException:
            self.broken = True
            raise
        return record

    def consume(self, key, *, clock_id, speaker, through, purpose="engineering_boundary"):
        head = self.retained()["head"]
        need(purpose == "engineering_boundary", "ADMISSION_REQUIRED")
        self._checked(key)
        ep = self.memory.episodes[key]
        need(ep["claim_kind"] == "derived_relation", "MOMENT_REQUIRED")
        for parent in ep["evidence"]:
            self._checked(parent)
        need(clock_id == ep["source"]["clock_id"], "CLOCK")
        need(speaker == ep["content"]["who"], "SPEAKER")
        need(rational(through) >= rational(ep["event_time"]["available_after"]), "FUTURE_ANNOTATION")
        result = self.evidence.views[ep["source"]["segment"]].read()
        need(self.retained()["head"] == head, "STALE_CONSUMER")
        return dict(head=head, moment=json.loads(canonical(ep)), **result)

    def snapshot(self):
        return dict(retained=self.retained(), memory=self.memory.snapshot(), flags=dict(FLAGS))

    def close(self):
        self.broken = True
        self.journal.close()
