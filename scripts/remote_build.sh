#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: remote_build.sh --remote yes|no --build-cmd "<command>" [--host user@host] [--threads N]

When --remote yes:
  --host is required (SSH target, e.g. user@remote)
  Docker image is derived from .vscode/CONTAINER_REPO_NAME and .vscode/CONTAINER_VERSION_TOP

When --remote no:
  The build command runs locally; --host is ignored.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

remote_build=""
build_cmd=""
remote_host=""
threads=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote=*)
            remote_build="${1#--remote=}"
            if [[ -z "${remote_build}" ]]; then
                die "--remote requires a value: yes or no."
            fi
            shift
            ;;
        --remote)
            if [[ $# -lt 2 || "${2}" == -* || -z "${2}" ]]; then
                die "--remote requires a value: yes or no."
            fi
            remote_build="$2"
            shift 2
            ;;
        --build-cmd=*)
            build_cmd="${1#--build-cmd=}"
            if [[ -z "${build_cmd}" ]]; then
                die "--build-cmd requires a non-empty command string."
            fi
            shift
            ;;
        --build-cmd)
            if [[ $# -lt 2 || "${2}" == -* || -z "${2}" ]]; then
                die "--build-cmd requires a non-empty command string."
            fi
            build_cmd="$2"
            shift 2
            ;;
        --host=*)
            remote_host="${1#--host=}"
            shift
            ;;
        --host)
            if [[ $# -lt 2 ]]; then
                die "--host requires a value (use --host \"\" for local builds with no host)."
            fi
            if [[ "${2}" == -* ]]; then
                # Some task runners drop empty prompt-string values, yielding:
                #   --host --build-cmd ...
                # Treat this as an intentionally empty host and continue parsing.
                remote_host=""
                shift 1
                continue
            fi
            remote_host="$2"
            shift 2
            ;;
        --threads=*)
            threads="${1#--threads=}"
            if [[ -z "${threads}" ]]; then
                die "--threads requires a positive integer."
            fi
            shift
            ;;
        --threads)
            if [[ $# -lt 2 || "${2}" == -* || -z "${2}" ]]; then
                die "--threads requires a positive integer."
            fi
            threads="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            die "Unknown argument: $1"
            ;;
    esac
done

if [[ -z "${remote_build}" || -z "${build_cmd}" ]]; then
    usage
    die "Both --remote and --build-cmd are required."
fi

if [[ "${remote_build}" != "yes" && "${remote_build}" != "no" ]]; then
    die "Invalid --remote value: ${remote_build} (expected yes|no)."
fi

if [[ -n "${threads}" ]]; then
    if ! [[ "${threads}" =~ ^[0-9]+$ ]] || [[ "${threads}" -le 0 ]]; then
        die "Invalid --threads value: ${threads} (expected positive integer)."
    fi
    build_cmd="export CISTEM_BUILD_THREADS=${threads}; ${build_cmd}"
fi

# Normalize host input:
# - trim all whitespace (a "blank-looking" prompt default may be a space)
# - treat common local sentinels as empty host
remote_host="$(printf '%s' "${remote_host}" | tr -d '[:space:]')"
remote_host_lower="$(printf '%s' "${remote_host}" | tr '[:upper:]' '[:lower:]')"
if [[ "${remote_host_lower}" == "no" || "${remote_host_lower}" == "none" || "${remote_host_lower}" == "local" || "${remote_host_lower}" == "skip" ]]; then
    remote_host=""
fi

if [[ "${remote_build}" == "yes" && -z "${remote_host}" ]]; then
    echo "Remote host is blank; falling back to local build." >&2
    remote_build="no"
fi

if [[ "${remote_build}" == "no" ]]; then
    bash -lc "${build_cmd}"
    exit $?
fi

# --- Remote build path ---

# Determine workspace root
workspace_root="${PWD}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${workspace_root}/.clang-format" && -f "${script_dir}/../.clang-format" ]]; then
    workspace_root="$(cd "${script_dir}/.." && pwd)"
fi

# Validate --host
if [[ -z "${remote_host}" ]]; then
    die "--host is required for remote builds (e.g. --host user@remote)."
fi

# Derive Docker image from version-controlled config files
repo_file="${workspace_root}/.vscode/CONTAINER_REPO_NAME"
version_file="${workspace_root}/.vscode/CONTAINER_VERSION_TOP"

if [[ ! -f "${repo_file}" ]]; then
    die "Cannot find ${repo_file} — is .vscode linked to a config directory?"
fi
if [[ ! -f "${version_file}" ]]; then
    die "Cannot find ${version_file} — is .vscode linked to a config directory?"
fi

remote_image="$(tr -d '[:space:]' < "${repo_file}"):v$(tr -d '[:space:]' < "${version_file}")"
echo "Remote build: host=${remote_host}, image=${remote_image}" >&2

if ! command -v ssh >/dev/null 2>&1; then
    die "ssh is not available on this system."
fi

ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)

if ! ssh "${ssh_opts[@]}" "${remote_host}" "true" >/dev/null 2>&1; then
    echo "Unable to connect to ${remote_host} with passwordless SSH." >&2
    echo "This usually means SSH keys are not set up for this host." >&2
    if command -v ssh-copy-id >/dev/null 2>&1; then
        printf "Would you like to run ssh-copy-id to set up key auth? [y/N] " >&2
        read -r answer
        if [[ "${answer}" =~ ^[Yy] ]]; then
            echo "Running: ssh-copy-id ${remote_host}" >&2
            ssh-copy-id "${remote_host}" || die "ssh-copy-id failed."
            # Verify it worked
            if ! ssh "${ssh_opts[@]}" "${remote_host}" "true" >/dev/null 2>&1; then
                die "SSH key setup succeeded but passwordless login still fails for ${remote_host}."
            fi
            echo "SSH key auth configured successfully." >&2
        else
            die "Cannot proceed without SSH access to ${remote_host}."
        fi
    else
        die "ssh-copy-id not found. Manually copy your SSH public key to ${remote_host}."
    fi
fi

remote_workspace_quoted="$(printf '%q' "${workspace_root}")"
if ! ssh "${ssh_opts[@]}" "${remote_host}" "test -d ${remote_workspace_quoted}" >/dev/null 2>&1; then
    die "Remote host missing workspace path: ${workspace_root}"
fi

if ! ssh "${ssh_opts[@]}" "${remote_host}" "command -v docker >/dev/null 2>&1"; then
    die "Docker is not available on remote host (${remote_host})."
fi

remote_path_exists() {
    local path="$1"
    ssh "${ssh_opts[@]}" "${remote_host}" "test -e $(printf '%q' "${path}")" >/dev/null 2>&1
}

# Run as the remote user's uid/gid so file permissions match the host filesystem
remote_uid=$(ssh "${ssh_opts[@]}" "${remote_host}" "id -u")
remote_gid=$(ssh "${ssh_opts[@]}" "${remote_host}" "id -g")

docker_cmd=(docker run --rm --network host)
docker_cmd+=(--user "${remote_uid}:${remote_gid}")
docker_cmd+=(-v "${workspace_root}:${workspace_root}")
docker_cmd+=(-v "${workspace_root}:/sa_shared/git/cisTEM")
docker_cmd+=(-w "${workspace_root}")
if remote_path_exists "/scratch"; then
    docker_cmd+=(-v "/scratch:/scratch")
fi

if remote_path_exists "/sa_shared"; then
    docker_cmd+=(-v "/sa_shared:/sa_shared")
fi

if [[ -n "${DISPLAY:-}" ]] && remote_path_exists "/tmp/.X11-unix"; then
    docker_cmd+=(-e "DISPLAY=${DISPLAY}")
    docker_cmd+=(-v "/tmp/.X11-unix:/tmp/.X11-unix")
fi

if [[ -n "${XAUTHORITY:-}" ]] && remote_path_exists "${XAUTHORITY}"; then
    docker_cmd+=(-v "${XAUTHORITY}:/home/cisTEMdev/.Xauthority")
fi

docker_cmd+=("${remote_image}" bash -lc "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; ${build_cmd}")

remote_cmd="$(printf '%q ' "${docker_cmd[@]}")"
remote_cmd="${remote_cmd% }"

ssh "${ssh_opts[@]}" "${remote_host}" "${remote_cmd}"
