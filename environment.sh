#!/bin/bash

set -e # exit on error

# Initialize conda if not already initialized
for conda_path in "/ihme/homes/sbachmei/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/usr/local/miniconda3" "/usr/local/anaconda3"; do
  if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
    echo
    echo "Initializing conda from $conda_path"
    source "$conda_path/etc/profile.d/conda.sh"
    break
  else
    echo
    echo "ERROR: Unable to find conda in expected locations."
    exit 1
  fi
done

# Reset OPTIND so help can be invoked multiple times per shell session.
OPTIND=1
Help()
{ 
   # Display Help
   echo
   echo "Script to automatically create and validate conda environments."
   echo
   echo "Syntax: source environment.sh [-h|t|v]"
   echo "options:"
   echo "h     Print this Help."
   echo "t     Type of conda environment. Either 'simulation' (default) or 'artifact'."
   echo "f     Force creation of a new environment."
   echo "l     Install git lfs."
}

# Define variables
username=$(whoami)
env_type="simulation"
make_new="no"
install_git_lfs="no"

# Process input options
while getopts ":hflt:" option; do
   case $option in
      h) # display help
         Help
         exit 0;;
      t) # Type of conda environment to build
         env_type=$OPTARG;;
      f) # Force creation of a new environment
         make_new="yes";;
      l) # Install git lfs
         install_git_lfs="yes";;
     \?) # Invalid option
         echo
         echo "ERROR: Invalid option"
         exit 1;;
   esac
done

# Parse environment name
env_name=$(basename "`pwd`")
env_name+="_$env_type"
branch_name=$(git rev-parse --abbrev-ref HEAD)
# Determine which requirements.txt to install from
if [ $env_type == 'simulation' ]; then
  install_file="requirements.txt"
elif [ $env_type == 'artifact' ]; then
  install_file="artifact_requirements.txt"
else
  echo
  echo "Invalid environment type. Valid argument types are 'simulation' and 'artifact'."
  exit 1 
fi

# Pull repo to get latest changes from remote if remote exists
git ls-remote --exit-code --heads origin $branch_name >/dev/null 2>&1
exit_code=$?
if [[ $exit_code == '0' ]]; then
  git fetch --all
  echo
  echo "Git branch '$branch_name' exists in the remote repository, pulling latest changes..."
  git pull origin $branch_name
fi

# Check if environment exists already
create_env=$(conda info --envs | grep $env_name | head -n 1)
if [[ $create_env == '' ]]; then
  # No environment exists with this name
  echo
  echo "Environment $env_name does not exist."
  create_env="yes"
  env_exists="no"
elif [[ $make_new == 'yes' ]]; then
  # User has requested to make a new environment
  echo
  echo "Making a new environment."
  create_env="yes"
  env_exists="yes"
else
  env_exists="yes"
  conda activate $env_name
  # Check if existing environment needs to be recreated
  echo
  echo "Existing environment found for $env_name."
  one_week_ago=$(date -d "7 days ago" '+%Y-%m-%d %H:%M:%S')
  creation_time="$(head -n1 $CONDA_PREFIX/conda-meta/history)"
  creation_time=$(echo $creation_time | sed -e 's/^==>\ //g' -e 's/\ <==//g')
  requirements_modification_time="$(date -r $install_file '+%Y-%m-%d %H:%M:%S')"
  # Check if existing environment is older than a week or if environment was built 
  # before last modification to requirements file. If so, mark for recreation.
  if [[ $creation_time < $one_week_ago ]] || [[ $creation_time < $requirements_modification_time ]]; then
    echo
    echo "Environment is stale. Deleting and remaking environment..."
    create_env="yes"
  else
    # Install json parser if it is not installed
    jq_exists=$(conda list | grep -w jq || true)
    if [[ $jq_exists == '' ]]; then
      # Empty string is no return on grep
      conda install jq -c anaconda -y
    fi
    # Check if there has been an update to vivarium packages since last modification to requirements file
    # or more reccent than environment creation
    # Note: The lines we will return via grep will look like 'vivarium>=#.#.#' or will be of the format 
    # 'vivarium @ git+https://github.com/ihmeuw/vivarium@SOME_BRANCH'
    framework_packages=$(grep -E '^[^#]*vivarium|^[^#]*gbd|^[^#]*risk_distribution|^[^#]*layered_config' $install_file)
    num_packages=$(grep -E '^[^#]*vivarium|^[^#]*gbd|^[^#]*risk_distribution|^[^#]*layered_config' -c $install_file)
    
    echo
    echo "Checking framework packages are up to date:"
    for pkg in $framework_packages; do
      echo "  - $pkg"
    done

    # FIXME: need to check artifactory - not github - for vivarium_gbd_access and vivarium_cluster_tools!


    # Iterate through each return of the grep output
    for ((i = 1; i <= $num_packages; i++)); do
      line=$(echo "$framework_packages" | sed -n "${i}p")
      # Check if the line contains '@'
      if [[ "$line" == *"@"* ]]; then
          repo_info=(${line//@/ })
          repo=${repo_info[0]}
          repo_branch=${repo_info[2]}
          curl_response=$(curl -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/ihmeuw/$repo/commits?sha=$repo_branch 2>/dev/null)
          if [[ -z "$curl_response" || "$curl_response" == "null" ]]; then
              echo
              echo "WARNING: Could not fetch update information for $repo. Skipping update check."
              continue
          fi
          last_update_time=$(echo "$curl_response" | jq -r '.[0].commit.committer.date // empty' 2>/dev/null)
      else
          repo=$(echo "$line" | cut -d '>' -f1)
          curl_response=$(curl -s https://pypi.org/pypi/$repo/json 2>/dev/null)
          if [[ -z "$curl_response" || "$curl_response" == "null" ]]; then
              echo
              echo "WARNING: Could not fetch update information for $repo. Skipping update check."
              continue
          fi
          # Check if the response contains an error message (package not found)
          if echo "$curl_response" | jq -e '.message' >/dev/null 2>&1; then
              echo
              echo "WARNING: Package $repo not found on PyPI. Skipping update check."
              continue
          fi
          last_update_time=$(echo "$curl_response" | jq -r '.releases | to_entries | max_by(.key) | .value | .[0].upload_time // empty' 2>/dev/null)
      fi
      
      if [[ -z "$last_update_time" || "$last_update_time" == "null" ]]; then
          echo
          echo "WARNING: Could not parse update time for $repo. Skipping update check."
          continue
      fi
      
      # Try to parse the date, skip if it fails
      parsed_date=$(date -d "$last_update_time" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
      if [[ $? -ne 0 || -z "$parsed_date" ]]; then
          echo
          echo "WARNING: Could not parse date '$last_update_time' for $repo. Skipping update check."
          continue
      fi
      
      if [[ $creation_time < $parsed_date ]]; then
        create_env="yes"
        echo
        echo "Last update time for $repo: $parsed_date. Environment is stale. Remaking environment..."
        break
      fi
    done
  fi
fi

if [[ $create_env == 'yes' ]]; then
  if [[ $env_exists == 'yes' ]]; then
    if [[ $env_name == $CONDA_DEFAULT_ENV ]]; then
      conda deactivate
    fi
    conda remove -n $env_name --all -y
  fi
  # Create conda environment
  conda create -n $env_name python=3.11 -c anaconda -y
  conda activate $env_name
  # NOTE: update branch name if you update requirements.txt in a branch
  echo
  echo "Installing packages for $env_type environment"
  pip install uv
  artifactory_url="https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple"
  uv pip install -r $install_file --extra-index-url $artifactory_url --index-strategy unsafe-best-match
  # Editable install of repo
  uv pip install -e .[dev] --extra-index-url $artifactory_url --index-strategy unsafe-best-match
  # Install redis for simulation environments
  if [ $env_type == 'simulation' ]; then
    conda install redis -c anaconda -y
  fi
  # Install git lfs if requested
  if [ $install_git_lfs == 'yes' ]; then
    git lfs install
  fi
else
  echo
  echo "Existing environment validated"
fi

echo
echo "*** FINISHED ***"
echo
echo "Don't forget to activate the environment:"
echo "conda activate $env_name"
