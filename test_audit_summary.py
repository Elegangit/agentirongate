import csv
import json

import pytest

from gatebench.audit import RATING_FIELDS, agreement, read_ratings, summarize_audit
from gatebench.io import write_json


def sheet(path, rater, labels):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'rater', *RATING_FIELDS, 'notes'])
        writer.writeheader()
        for case, label in labels.items():
            writer.writerow({'case_id': case, 'rater': rater,
                             **{f: label for f in RATING_FIELDS},
                             'notes': 'Insufficient evidence' if label == 'uncertain' else ''})
    return path


def test_agreement_degenerate_and_disagreement():
    assert agreement(['yes'], ['yes'])['cohens_kappa'] is None
    result = agreement(['yes', 'no'], ['no', 'yes'])
    assert result['raw_agreement'] == 0 and result['cohens_kappa'] == -1
    assert agreement([], [])['raw_agreement'] is None


@pytest.mark.parametrize('labels', [{'a': ''}, {'a': 'maybe'}, {'x': 'yes'}, {}])
def test_incomplete_or_invalid_sheets_rejected(tmp_path, labels):
    with pytest.raises(ValueError):
        read_ratings(sheet(tmp_path / 'r.csv', 'A', labels), {'a'})


def test_duplicate_cases_rejected(tmp_path):
    path = sheet(tmp_path / 'r.csv', 'A', {'a': 'yes'})
    with path.open('a') as stream:
        stream.write('a,A,yes,yes,yes,yes,\n')
    with pytest.raises(ValueError, match='duplicate'):
        read_ratings(path, {'a'})


def test_summary_counts_exclusions_and_disagreements(tmp_path):
    audit = tmp_path / 'audit'
    ids = ['a', 'b', 'c', 'd']
    write_json(audit / 'rater_cases.json', {'evidence': 'scripted_fixture_only',
                                         'cases': [{'case_id': c} for c in ids]})
    write_json(audit / 'PRIVATE_mapping.json', [
        {'case_id': c, 'model_id': 'fixture', 'condition': 'minimal',
         'metrics': {'completion': auto, **before}}
        for c, auto, before in [('a', 1, {'completed_before_checkpoint': 0}),
                               ('b', 0, {'completed_before_checkpoint': 0}),
                               ('c', 1, {'completed_before_checkpoint': 1}),
                               ('d', 1, {})]])
    left = sheet(tmp_path / 'left.csv', 'A', {'a': 'no', 'b': 'yes', 'c': 'yes', 'd': 'uncertain'})
    right = sheet(tmp_path / 'right.csv', 'B', {c: 'yes' for c in ids})
    dest = summarize_audit(audit, left, right, tmp_path / 'report')
    report = json.loads((dest / 'audit_summary.json').read_text())
    result = report['automatic_vs_human'][1]
    assert result['scope'] == 'recovery'
    assert result['false_positive'] == 1 and result['false_negative'] == 1
    assert result['precompleted_excluded'] == result['unknown_checkpoint_excluded'] == 1
    assert report['needs_adjudication'] == ['a', 'd']
    assert report['evidence'] == 'scripted_fixture_only'
    with pytest.raises(FileExistsError):
        summarize_audit(audit, left, right, dest)
    with pytest.raises(ValueError, match='distinct'):
        summarize_audit(audit, left, left, tmp_path / 'other')
