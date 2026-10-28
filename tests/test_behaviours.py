from behaviour_prediction.monitor.behaviours import checkpoint_tags


def test_uncertain_response_does_not_establish_tag_absence():
    values = checkpoint_tags([
        dict(claude_harmful="1", claude_behaviour_tags="harassment"),
        dict(claude_harmful="", claude_behaviour_tags="")])
    assert values["harassment"] == 1
    assert values["hate_speech"] is None


def test_known_nonharmful_response_has_no_behaviour_tags():
    values = checkpoint_tags([dict(claude_harmful="0", claude_behaviour_tags="")])
    assert set(values.values()) == {0}
