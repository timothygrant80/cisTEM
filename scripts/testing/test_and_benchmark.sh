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

# PARAMETERS

MOVIE_FILE=/tmp/hiv_images_shift_noise_80x80x10.mrc
PIXEL_SIZE=1.0
VOLTAGE=300
SPHERICAL_ABBERATION=2.7
AMPLITUDE_CONTRAST=0.07
BOX_SIZE=512
BINNING_FACTOR=1

# UNBLUR - No expert options

$cisTEM_path/unblur <<EOF
$MOVIE_FILE
/tmp/align.mrc
/tmp/shifts.txt
$PIXEL_SIZE
$BINNING_FACTOR
yes                              # Apply exposure Filter
$VOLTAGE
1.0                              # Exposure per frame
0.0                              # Pre-exposure
no                               # Expert options
no                               # Mag distortion correction
1                                # Number of threads
EOF

# CTFFIND - Movie, No expert options, No phase, No Tilt

$cisTEM_path/ctffind <<EOF
$MOVIE_FILE
yes                              # Is movie
2                                # Frames to average
/tmp/diagnostic_output.mrc         
$PIXEL_SIZE
$VOLTAGE
$SPHERICAL_ABBERATION
$AMPLITUDE_CONTRAST
$BOX_SIZE
30.0                             # Low resolution limit
5.0                              # High resolution limit
5000.0                           # Low defocus search range
50000.0                          # High defocus search range
100.0                            # Defocus step during search
no                               # Astigmatism known
no                               # Slow, exhaustive, search
no                               # Restrain astigmatism
no                               # Find phase shift     
no                               # Find sample tilt
no                               # Set expert options
EOF

