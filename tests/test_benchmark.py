"""M4 harness tests — verify the benchmark ORCHESTRATES and SCORES correctly,
using deterministic fake models (no GPU). The real quality numbers come from
running it on falcon; here we prove the harness itself is sound.
"""

from ctx.benchmark import format_report, run_benchmark


class GoodFake:
    """A 'capable' model: digest = active bodies; answers the judge correctly."""

    def initial(self):
        return ()

    def fold(self, state, entry):
        return state + (entry["body"],)

    def render(self, state):
        return " | ".join(state)

    def generate(self, prompt, max_new_tokens=None):
        return ("yes — both describe caching repeated quote lookups. "
                "merged: request timeout 8s; cache TTL 60 seconds; backoff exponential.")


class BadFake(GoodFake):
    """A 'weak' model: rambles, misses the overlap and the facts."""

    def generate(self, prompt, max_new_tokens=None):
        return "hmm, here is some unrelated text about the weather."


def test_harness_runs_and_scores_a_capable_model():
    r = run_benchmark(GoodFake(), n_stream=6)
    assert set(r) == {"high_freq", "inverted", "duplicate", "merge"}
    assert r["high_freq"]["n_events"] == 6
    assert r["high_freq"]["events_per_s"] is not None
    # #3 / #4 scoring picks up a model that answers correctly
    assert r["duplicate"]["flagged_duplicate"] is True
    assert r["merge"]["recall"] == 1.0
    assert format_report(r).startswith("=== M4 BENCHMARK ===")


def test_harness_scores_a_weak_model_low():
    r = run_benchmark(BadFake(), n_stream=4)
    assert r["duplicate"]["flagged_duplicate"] is False
    assert r["merge"]["recall"] == 0.0


def test_inverted_holds_regardless_of_model():
    """#2 is architectural: the store drops the superseded fact, so BOTH a strong
    and a weak model pass — retraction never depends on model intelligence."""
    for fake in (GoodFake(), BadFake()):
        r = run_benchmark(fake, n_stream=2)
        assert r["inverted"]["reverted_fact_absent"] is True
        assert r["inverted"]["new_fact_present"] is True
