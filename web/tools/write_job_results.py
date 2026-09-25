#!/usr/bin/env python3
"""Write a finished job's results to the project database again.

    cd server && python3 ../tools/write_job_results.py <project id> <job id>

For a job whose tasks all finished but whose log ends in "could not write
results to the project database: ...": the tasks' results are still stored
(JOB_TASKS), so once the cause is fixed the stage adapter's finalize() can be
run again without re-running the job. Refuses a job that is not completed,
and one whose results are already written (metrics carry the write summary).
Run it on the server machine, from server/ (the same data directory); the
server may stay up. The project id is the directory name under
server/data/projects/, the job id the one shown under the job's name.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import db  # noqa: E402
import cistem_server  # noqa: E402


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    project_id, job_id = argv[1], argv[2]
    if not (db.PROJECTS_ROOT / project_id / "project.db").is_file():
        print("no project {} under {}".format(project_id, db.PROJECTS_ROOT))
        return 1
    conn = db.get_conn(project_id)
    try:
        row = conn.execute("SELECT * FROM JOBS WHERE JOB_ID=?", (job_id,)).fetchone()
        if row is None:
            print("no job {} in project {}".format(job_id, project_id))
            return 1
        if row["STATUS"] != "completed":
            print("job {} is {}, not completed; only a completed job's results can be written".format(job_id, row["STATUS"]))
            return 1
        if row["STAGE"] not in cistem_server.stages.ADAPTERS:
            print("job {} is a {} job, which has no results to write this way".format(job_id, row["STAGE"]))
            return 1
        metrics = json.loads(row["METRICS_JSON"]) if row["METRICS_JSON"] else {}
        if any(k.endswith("_written") for k in metrics):
            print("job {} already has its results written: {}".format(job_id, metrics))
            return 1
        summary = cistem_server.write_job_results(conn, project_id, row)
        if summary is None:
            print("the write failed again; the job's log says why")
            return 1
        metrics.update(summary)
        with conn:
            conn.execute("UPDATE JOBS SET METRICS_JSON=? WHERE JOB_ID=?", (json.dumps(metrics), job_id))
        print("wrote results of job {}: {}".format(job_id, summary))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
