"""`merge2d`, the other leg of a 2D classification round.

After a round's refine2d tasks have each dumped their class sums, one
merge2d task adds the dumps up and writes the round's class averages
(ClassificationManager::RunMerge2dJob(), format "tti": output class
averages, dump-file seed, number of dump files). Submitted by
server/classification.py as a hidden child job with STAGE "class2d_merge2d";
this module only tells the runner which program that is.
"""

PROGRAM = {"name": "merge2d", "executable": "merge2d"}

ARGUMENT_NAMES = ("output_class_averages", "dump_file_seed", "number_of_dump_files")
ARGUMENT_TYPES = "tti"


def finalize(conn, project_id, job, sent_tasks, task_rows, log):
    return {"merge2d_tasks_ok": sum(1 for r in task_rows if r["STATUS"] == "ok")}


def describe_summary(summary):
    return "class averages merged" if summary.get("merge2d_tasks_ok") else "merge2d wrote nothing"
