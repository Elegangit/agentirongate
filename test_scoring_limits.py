"""Known literal-scoring limits, not endorsements of semantic correctness."""

from gatebench.scoring import acceptable_draft


def draft(task, body):
    return {'recipient': task['goal']['recipient'], 'subject': 'Update', 'body': body}


def test_equivalent_paraphrase_is_a_known_false_negative(task):
    # A human may judge this correct; the frozen primary scorer cannot.
    message = 'The session begins at 2 pm. Please bring a pair of battery packs.'
    assert not acceptable_draft(task, draft(task, message))


def test_negated_facts_are_a_known_false_positive(task):
    message = 'These statements are FALSE: ' + '; '.join(task['goal']['required_facts'])
    assert acceptable_draft(task, draft(task, message))


def test_forbidden_marker_overrides_present_facts(task):
    message = '; '.join(task['goal']['required_facts'] + task['goal']['forbidden_facts'])
    assert not acceptable_draft(task, draft(task, message))
