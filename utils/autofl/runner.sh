LABEL_PREFIX=$1
if [ -z  "$1" ]; then
    echo "Please provide an experiment label."
    exit 0
fi
REPETITION=$2
DATASET=$3 # only supports defects4j - update prompt file to support tools for bugsinpy
MODEL=$4
PROMPT_FILE=$5
PROJECT=$6

DATA_DIR=./data/${DATASET}/
ENDPOINT="http://localhost:11434/api/generate" # ollama default endpoint
BUDGET="10"
NUM_TESTS="1"

trap 'echo interrupted; exit 1' INT

for rep in $(seq 1 "$REPETITION"); do
    label="${LABEL_PREFIX}${rep}"
    save_dir="results/${label}/${MODEL}"
    mkdir -p "${save_dir}"
    if [ -z "$PROJECT" ]; then
        bug_list=$(ls -d ${DATA_DIR}/*/ | xargs -n1 basename)
    else
        bug_list=$(ls -d ${DATA_DIR}/*/ | xargs -n1 basename | grep ${PROJECT})
    fi
    for bugname in $bug_list; do
        save_file="${save_dir}/XFL-${bugname}.json"
        if [ -f ${save_file} ]; then
            echo "${save_file} exists"
            continue
        fi
        if [ -f "${DATA_DIR}/${bugname}/snippet.json" ]; then
            cmd="python autofl.py -m ${MODEL} -e ${ENDPOINT} -b ${bugname} -p ${PROMPT_FILE} -o ${save_file} --max_budget ${BUDGET} --max_num_tests ${NUM_TESTS} --show_line_number --postprocess_test_snippet --allow_multi_predictions --test_offset $((rep - 1)) --measure_power_consumption"
            # measure_power_consumption option only works when there are both pynvml module and GPU(s), disable otherwise
            echo ${cmd}
            ${cmd}
        fi
    done
done
