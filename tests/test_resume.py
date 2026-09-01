from savi.phase0 import job_key, pending_jobs


def _job(state, horizon, continuation):
    return {"state_id": state, "horizon": horizon, "continuation_id": continuation}


def test_resume_skips_exactly_completed_tuples():
    jobs = [_job("s", 0, 0), _job("s", 256, 0), _job("s", 256, 1), _job("t", 0, 0)]
    completed = [dict(jobs[0]), dict(jobs[2])]
    assert pending_jobs(jobs, completed) == [jobs[1], jobs[3]]


def test_job_key_normalizes_json_integer_fields():
    assert job_key({"state_id": "s", "horizon": 256, "continuation_id": 1}) == ("s", 256, 1)
