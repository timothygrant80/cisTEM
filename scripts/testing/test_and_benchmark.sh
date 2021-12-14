#!/bin/bash
#
# Basic cisTEM program benchmark
#
# Johannes Elferich, December 2021
#
set -e 
print_help() {
  cat <<DOC
   
Quick script to benchmark key cisTEM programs using embedded images.
Usage:
  ${_ME} [cisTEM bin path]
  ${_ME} -h | --help
Options:
  -h --help  Show this screen.
DOC
}

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}


msg() {
  echo >&2 -e "${1-}"
}

parse_params() {
  # default values of variables set from params
  flag=0
  param=''

  while :; do
    case "${1-}" in
    -h | --help) print_help ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  args=("$@")

  # check required params and arguments
  [[ ${#args[@]} -ne 1 ]] && die "cisTEM path not specified"

  cisTEM_path=${args[0]}
  return 0
}


parse_params "$@"

# $cisTEM_path/console_test

# TODO: Provide more realistic benchmark (Maybe montage hiv image together and simulate ctf(?))

$cisTEM_path/unblur <<EOF
/tmp/hiv_images_shift_noise_80x80x10.mrc
/tmp/align.mrc
/tmp/shifts.txt
1.0
1
yes
300.0
1.0
0.0
no
no
1
EOF

$cisTEM_path/ctffind <<EOF
/tmp/hiv_images_shift_noise_80x80x10.mrc
yes
2
/tmp/diagnostic_output.mrc
1.0
300.0
2.7
0.07
512
30.0
5.0
5000.0
50000.0
100.0
no
no
no
no
no
no
EOF

