import json
import os
from types import SimpleNamespace as NS
import pytest
from steward.ego.skills import Job, SkillExecutor
from steward.ego import receipt_log


def test_complete_public_receipt_append_and_disabled(monkeypatch, tmp_path):
    job = Job('job-test', 'seat_screw', 'r', state='failed', started=2., finished=3.,
              postcondition={k: {'value': [False, None, '原因']} for k in (
                  'seatDelivery', 'wristAlignment', 'timeoutCompletion', 'fastenerReturns',
                  'holeProgress', 'completionGate', 'postflipDestination', 'wristRelease')})
    monkeypatch.delenv('STEWARD_RECEIPT_LOG_DIR', raising=False)
    receipt_log.append(job, {}, 10.)
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv('STEWARD_RECEIPT_LOG_DIR', str(tmp_path))
    for _ in range(2):
        receipt_log.append(job, {'params': {'hole': 'selected'}}, 10.)
    rows = [json.loads(s) for s in (tmp_path / f'receipts-{os.getpid()}.jsonl').read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]['receipt'] == job.to_dict()
    assert rows[0]['params'] == {'hole': 'selected'}
    assert rows[0]['simTime'] == {'started': 2., 'finished': 3.}
    monkeypatch.setenv('STEWARD_RECEIPT_LOG_DIR', str(tmp_path / 'missing' / '..' / 'bad'))
    monkeypatch.setattr(receipt_log.Path, 'mkdir', lambda *a, **kw: (_ for _ in ()).throw(OSError('disk full')))
    receipt_log.append(job, {}, 10.)
    assert job.state == 'failed'


@pytest.mark.parametrize('raises', [False, True])
def test_wrapper_logs_after_all_finish_paths(monkeypatch, raises):
    seen=[]
    job=Job('j','look_around','r')
    def run(*args):
        job.postcondition={'finishHook':True}
        if raises: raise RuntimeError('finish failed')
    monkeypatch.setattr(receipt_log,'append',lambda j,r,t:seen.append(j.postcondition.copy()))
    fake=NS(_run_job=run)
    if raises:
        with pytest.raises(RuntimeError): SkillExecutor._run(fake,job,job.skill,{})
    else: SkillExecutor._run(fake,job,job.skill,{})
    assert seen==[{'finishHook':True}]
