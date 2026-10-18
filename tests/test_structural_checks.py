import pytest

from gatebench.providers import OfflineProvider
from gatebench.runner import collect_prefix, continue_branch
from gatebench.tasks import load_tasks


@pytest.mark.parametrize('index,reads,drafts,label', [
    (0, ['notes', 'schedule'], 0, 'public'),
    (1, ['notes', 'bulletin'], 0, 'public'),
    (2, ['notes', 'budget'], 1, 'confidential'),
])
def test_structural_checkpoint_and_scripted_recovery(root, cfg, index, reads, drafts, label):
    task = load_tasks(root / 'data/structural_checks.json')['tasks'][index]
    prefix = collect_prefix(task, OfflineProvider(task), cfg)
    assert prefix.termination == 'eligible'
    assert prefix.state.reads == reads
    assert len(prefix.state.drafts) == drafts and prefix.state.label == label
    assert not prefix.state.outbox
    for condition in ('minimal', 'rule', 'directed', 'hint'):
        result = continue_branch(task, prefix.checkpoint, condition, OfflineProvider(task), cfg)
        assert result['metrics']['new_completion'] == 1
        assert result['metrics']['unauthorized_effect'] == 0
        assert result['state']['label'] == label
        assert len(result['state']['outbox']) == 1
        assert result['state']['outbox'][0]['recipient'] == task['goal']['recipient']
    # Continuing branches does not mutate the shared parent.
    assert not prefix.state.outbox and len(prefix.state.drafts) == drafts
