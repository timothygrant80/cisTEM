#!/usr/bin/env python3
"""Write the checkout's version -- commit, commit date, branch -- as JSON for an
installed server to report.

    write_version.py [--repo DIR] OUT

`/api/version` runs git in the server's own directory, which works in a
checkout but not under share/cistem/web/server, where `make install` puts the
server; so the install writes `server/version.json` beside it from the
checkout it was installed from (the install-data-hook in Makefile.am), and the
route reads that file when git has nothing to say. Fields git cannot supply
(not a checkout, git not installed) are written as null and the exit status is
still 0: an install from a tarball must not fail over a decoration.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys


# This runs under whatever python3 the machine that installs has -- a CentOS 7 compile server has
# 3.6 -- so subprocess.run with PIPEs and universal_newlines, not the 3.7 keyword arguments for the same.
FAILURES = []   # why git had nothing to say, for the message at the end


def git(repo, args):
    try:
        result = subprocess.run(["git"] + args, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=10)
    except Exception as exc:  # noqa: BLE001 -- git missing, not executable, ...
        FAILURES.append("git {}: {}: {}".format(" ".join(args), type(exc).__name__, exc))
        return None
    if result.returncode != 0:
        FAILURES.append("git {} (exit {}): {}".format(" ".join(args), result.returncode, " ".join((result.stderr or "").split()) or "no message"))
        return None
    return result.stdout.strip() or None


def version_of(repo):
    """The same fields `/api/version` reports from a checkout."""
    commit = git(repo, ["rev-parse", "--short", "HEAD"])
    if commit and git(repo, ["status", "--porcelain"]):
        commit += "-dirty"
    return {
        "commit": commit,
        "commit_datetime": git(repo, ["log", "-1", "--format=%ci"]),
        "branch": git(repo, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        help="the checkout to describe (default: the one this script is in)")
    parser.add_argument("out", help="the JSON file to write")
    args = parser.parse_args(argv)
    info = version_of(args.repo)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    if info["commit"] is None:
        print("write_version: git could not describe {}; wrote {} with no version. The reason: {}".format(
            os.path.abspath(args.repo), args.out, "; ".join(FAILURES[:1]) or "unknown"), file=sys.stderr)
    else:
        print("wrote {}: {} ({})".format(args.out, info["commit"], info["branch"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
