from retailops.agent.policies import PolicyIndex


def test_policy_index_returns_relevant_cancellation_policy(policies: PolicyIndex) -> None:
    matches = policies.search("Can I cancel a shipped order?")

    assert matches
    assert any("cancel" in match.text.lower() for match in matches)
    assert matches[0].score > 0
