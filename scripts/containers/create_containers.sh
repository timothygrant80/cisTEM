#!/bin/bash

#########################################################################################################################################################################################
### PURPOSE: 
#####################
# To encourage the most unified build environment, we are using a two layered container system.
# The base image contains all of the dependencies that are not likely to change often, and are expensive to build/compile or even download. This should be the same for all devs.
# The full image is built on top of the base image and contains the dependencies that are more likely to change, and are less expensive to build/compile. This *may* be different for each dev.
#     NOTE: ideally, we all use the same build environment, but at least this way, any differences are a) well defined and b) easy to test/reproduce for other devs.
#########################################################################################################################################################################################

#########################################################################################################################################################################################
### USAGE:
#####################
# This script is intended to be run from the cisTEM/scripts/containers directory
# Based on the first arg, it will build the base[top] image, and tag it with the version number from the CONTAINER_VERSION_BASE file in the .vscode directory, which must by linked to your own .vscode_shared/YOUR_NAME
#    NOTE: When building the top layer, there is no check included that you have the correct baselayer built. We'll let docker handle that.
#########################################################################################################################################################################################

#########################################################################################################################################################################################
### NAME CONVENTION:
#####################
# Using the convention that the the tag is v<version> for the complete (base + top) and base_image_v<version> for the base image
#########################################################################################################################################################################################

usr_path="../../.vscode"

# Check for -h or --help
if [[ $1 == "-h" || $1 == "--help" ]] ; then
    echo ""
    echo "Usage: build_base.sh <base|top> [--no-cache] [--no-cuda] [--wx-version=<stable|dev>] [--compiler=<icpc|g++>] [--build-type=<static|dynamic>] [--npm ] [--ref-images ]"
    echo "      --no-cache: build without cache, must be second arg"
    echo ""
    echo "  positional args are optional and only affect the top layer build"
    echo "      --wx-version: stable or dev, default is stable (currently 3.0.5)"
    echo "      --compiler: icpc or g++, default is icpc [g++ builds not supported yet]"
    echo "      --build-type: static or dynamic, default is static [BUT only dynamic is supported for --wx-version dev]"
    echo "      --npm: build npm, default is false if not specified"
    echo "      --ref-images: build reference images, default is false if not specified"
        echo ""
    echo "For example, to build the base image without cache, and the top image with wxWidgets 3.1.5, g++, dynamic, npm, and ref-images:"
    echo "      build_base.sh base --no-cache --wx-version=dev --compiler=g++ --npm --ref-images"
    exit 0
fi

# Check the .vscode directory exists
if [[ ! -d ${usr_path} ]] ; then
    echo "This script must be run from the cisTEM/scripts/containers directory"
    echo "And the .vscode link must be established in the project root directory"
    exit 1
fi

# Check that we have "base" or "top" as an argument
#   base - this is the base image that should be the same for all devs working on cisTEM in order to ensure critical tool-chain compatibility
#   top - this is the top layer that is built on top of the base image and contains the dependencies that are more likely to change, and are less expensive to build/compile. This *may* be different for each dev.
if [[ $# -lt 1 ]] ; then
    echo "Usage: build_base.sh <base|top|-h|--help>"
    exit 1
else 
    if [[ $1 != "base" && $1 != "top" ]] ; then
        echo "Usage: build_base.sh <base|top|-h|--help>"
        exit 1
    else
        build_layer=$1
        # We shift here so we don't get stuck in the while loop for positional args
        shift
    fi
fi

# Check to see if we want to build without cache
if [[ $# -gt 0 ]] ; then
    if [[ $1 == "--no-cache" ]] ; then
        echo "Building without cache"
        skip_cache="--no-cache"
        # move the args down to see if we have positional args
        shift
    fi
else
    skip_cache=""
fi

# Now look for any additional arguments
# Default values are:
build_type="static"
build_compiler="icpc"
build_wx_version="stable"
build_npm="false"
build_ref_images="true"
build_pytorch="false"
build_ci_layer=""


while [[ $# -gt 0 ]]; do
  case $1 in
    --wx-version)
        build_wx_version="$2"
      # Check that the version is valid: stable or dev
        if [[ $build_wx_version != "stable" && $build_wx_version != "dev" && $build_wx_version != "gcc_ci" && $build_wx_version != "clang_ci" && $build_wx_version != "icpc_ci" ]] ; then
            echo "Invalid wx version: ($build_wx_version) - must be stable or dev (intel) or gcc_ci, clang_ci, or icpc_ci (for CI containers.)"
            exit 1
        fi
        shift # past argument
        shift # past value
        ;;
    --compiler)
        build_compiler="$2"
        # Check that the compiler is valid: icpc or g++
        if [[ $build_compiler != "icpc" && $build_compiler != "g++" ]] ; then
            echo "Invalid compiler: ($build-compiler) - must be icpc or g++"
            exit 1
        fi
        if [[ $build_compiler == "g++" ]] ; then
            echo "g++ builds not supported yet"
            exit 1
        fi
        shift # past argument
        shift # past value
        ;;
    --build-type)
        build_type="$2"
        # Check that the build type is valid: dynamic or static
        if [[ $build_type != "dynamic" && $build_type != "static" ]] ; then
            echo "Invalid build type: ($build-type) - must be dynamic or static"
            exit 1
        fi
        shift # past argument
        shift # past value
        ;;
    --npm)
        build_npm="true"
        shift # past argument
        ;;
    --ref-images)
        build_ref_images="$2"
        if [[ $build_ref_images != "true" && $build_ref_images != "false" ]] ; then
            echo "Invalid ref-images: ($build_ref_images) - must be true or false"
            exit 1
        fi
        shift # past argument
        shift # past value
        ;;
    --pytorch)
        build_pytorch="true"
        shift # past argument
        ;;
    -*|--*)
      echo "Unknown option $1, try running with -h for help."
      exit 1
      ;;
  esac
done

# Only dynamic builds are supported for wxWidgets dev, confirm this is true:
if [[ $build_wx_version == "dev" && $build_type == "static" ]] ; then
    echo "Dynamic builds are required for wxWidgets dev"
    exit 1
fi

if [[ $build_wx_version == "gcc_ci" || $build_wx_version == "clang_ci" || $build_wx_version == "icpc_ci" ]] ; then
    build_ci_layer="_${build_wx_version}"
fi

# These are effectively fixed constants.
# A copy with the updated version number will be placed in a temp file and used to build the container
path_to_base_dockerfile="base_image/"
path_to_top_dockerfile="top_image/"

# Get the wanted version number from the CONTAINER_VERSION file
# These are user specific and should be in the .vscode_shared/UserName directory
#   NOTE: in the future if more broadly adapted, we should have a .vscode_shared/NewUser (or something) that has good defaults and links to a cisTEM-org repo for pre-built containers
#         so that new users don't have to build from scratch.
if [[ ! -f ${usr_path}/CONTAINER_VERSION_BASE ]] ; then
    echo "CONTAINER_VERSION_BASE file not found"
    exit 1
fi
base_container_version=$(cat ${usr_path}/CONTAINER_VERSION_BASE)

if [[ ! -f ${usr_path}/CONTAINER_VERSION_TOP ]] ; then
    echo "CONTAINER_VERSION_TOP file not found"
    exit 1
fi
top_container_version=$(cat ${usr_path}/CONTAINER_VERSION_TOP)

# Also get the container repository information
# For example, on dockerhub, this would be username/reponame [bhimesbhimes/cistem_build_env]
if [[ ! -f ${usr_path}/CONTAINER_REPO_NAME ]] ; then
    echo "CONTAINER_REPOSITORY file not found"
    exit 1
else
    container_repository=$(cat ${usr_path}/CONTAINER_REPO_NAME)
fi


if [[ "x${build_layer}" == "xbase" ]] ; then
    echo "Building base layer with:"
    echo "    container version: ${base_container_version}"
    echo "    container repository: ${container_repository}"
    echo "    path to base dockerfile: ${path_to_base_dockerfile}"
    sleep 3
    prefix="base_image_v"
    container_version=${base_container_version}
    path_to_dockerfile=${path_to_base_dockerfile}
else
    prefix="v"
    container_version=${top_container_version}

    echo "Building top layer with:"
    echo "    wxWidgets version: ${build_wx_version}"
    echo "    compiler: ${build_compiler}"
    echo "    build type: ${build_type}"
    echo "    npm: ${build_npm}"
    echo "    ref-images: ${build_ref_images}"
    echo "    pytorch: ${build_pytorch}"
    echo "    container version: ${top_container_version}${build_ci_layer} "
    echo "    container base version: ${base_container_version}"
    echo "    container repository: ${container_repository}"
    echo "    path to top dockerfile: ${path_to_top_dockerfile}"
    # Modify the top_dockerfile to use the correct base image
    #   NOTE: there is no need to modify the base Docker file
    sleep 3
    path_to_dockerfile=$(mktemp -d)
    # Make sure we clean up the temp directoryman
    trap 'rm -rf -- "$path_to_dockerfile"' EXIT


    awk -v VER="base_image_v$base_container_version" -v REPO="FROM $container_repository" '{if($0 ~ "FROM fake_repo") print REPO":"VER; else print $0}' ${path_to_top_dockerfile}/Dockerfile > ${path_to_dockerfile}/Dockerfile
    cp ${path_to_top_dockerfile}/install*.sh ${path_to_dockerfile}/

    
    # Modify the devcontainer.json to use the correct full image, this should be soft linked from the project root to the .vscode_shared/UserName/devcontainer_VERSION.json
    # tmp_file=$(mktemp)
    # awk -v VER="v$top_container_version" -v REPO="    \"image\": \"$container_repository" '{if($0 ~ REPO) print REPO":"VER"\",";  else print $0}' ../../.devcontainer.json > $tmp_file
    # # awk -v VER="$container_version" '{if(/bhimesbhimes\/ibkr_tools/) print "   \"image\": \"bhimesbhimes/ibkr_tools:v"VER"\","; else print $0}' ${path_to_dockerfile}/../.devcontainer.json > tmp.json
    # cp --no-dereference $tmp_file ../../.devcontainer.json
    # rm $tmp_file
fi


# Print out the version and repository information
echo "Building ${container_repository}:${prefix}${container_version}${build_ci_layer} ${path_to_dockerfile}Dockerfile"


docker build ${skip_cache} --tag ${container_repository}:${prefix}${container_version}${build_ci_layer} \
    --build-arg build_type=${build_type} \
    --build-arg build_compiler=${build_compiler} \
    --build-arg build_wx_version=${build_wx_version} \
    --build-arg build_npm=${build_npm} \
    --build-arg build_ref_images=${build_ref_images} \
    --build-arg build_pytorch=${build_pytorch} \
    ${path_to_dockerfile}