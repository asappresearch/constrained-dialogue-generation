save_dir="results"
num_total_examples=1000
model_type="finetuned" #options = [pretrained, finetuned]
eval_type="real" #options = [real, simulated, both]
num_futures=1
num_fixed_keywords=5
num_fixed_generations=15
fixed_lambda=15
split="eval"
num_intermediate_save_steps=1
#nprocs=1

display_help () {
    echo "Usage: ${0} [option=value...]"
    echo
    echo "--DATA-DIR                 [Default=null]                                  Data directory"
    echo "--SAVE-DIR                 [Default=${save_dir}]                           Output directory for the results"
    echo "--RUN-APPROACHES           [Default=null]                                  Approaches to run"
    echo "--NUM-EVAL-EXAMPLES        [Default=${num_total_examples}]                 Number of eval examples to use"
    echo "--MODEL-TYPE               [Default=${model_type}]                         [pretrained, finetuned]"
    echo "--MODEL-PATH               [Default=null]                                  model_path if model-type is finetuned"
    echo "--EVAL-TYPE                [Default=${eval_type}]                          Evaluation type [real, simulated, both]"
    echo "--NUM-FUTURES              [Default=${num_futures}]                        Number of futures"
    echo "--NUM-KEYWORDS             [Default=${num_fixed_keywords}]                 Number of keywords"
    echo "--NUM-GENERATIONS          [Default=${num_fixed_generations}]              Number of generations"
    echo "--FIXED-LAMBDA             [Default=${fixed_lambda}]                       Fixed Lambda parameter"
    echo "--CONFIG-FILE              [Default=null]                                  Config file for the run"
    echo "--AGENT-MODEL-PATH         [Default=null]                                  Path to the trained agent model"
    echo "--KEYWORDS-FILE-PATH       [Default=null]                                  Keywords file path for data in the given split (default is eval/test data)"
    echo "--SPLIT                    [Default=eval]                                  Split in data"
    echo "--ONE-STEP-GEN             [Default=False]                                 One step generation"
    echo "--NUM-SAVE-STEPS           [Default=${num_intermediate_save_steps}]        Intermediate save steps"
    echo "--SIMULATION-DATASTORE     [Default=None]                                  Datastore created with batch size 1 for simulation"
}

declare -a run_approaches
declare -A args_to_conditions=( ["dbs"]="DirectedBeamSearch" \
                                ["fop"]="FuturesOfThePast" \
                                ["windowfop"]="WindowFuturesOfThePast" \
                                ["retrieve"]="Retrieval" \
                                ["prompt"]="Prompting" \
                                ["cgmh"]="CGMH" \
                                ["trainprompt"]="PromptingWithTraining" \
                                ["wfirst"]="WFirst" \
                                ["finetuned"]="FinetunedModel" \
                                ["none"]="")

for arg in "$@"
do
    key=$(echo $arg | cut -f1 -d=)
    val=$(echo $arg | cut -f2 -d=)

    case "$key" in
      --SAVE-DIR | --save-dir) save_dir=${val} ;;
      --RUN-APPROACHES | --run-approaches)
        i=0
        for approach in `echo "${val}" | sed "s/,/ /g"`;
        do
            run_approaches[i]=${args_to_conditions[$approach]}
            ((i=i+1))
        done
      ;;
      --NUM-EVAL-EXAMPLES | --num-eval-examples) num_total_examples=${val} ;;
      --MODEL-TYPE | --model-type) model_type=${val} ;;
      --MODEL-PATH | --model-path) model_path=${val} ;;
      --EVAL-TYPE | --eval-type) eval_type=${val} ;;
      --NUM-FUTURES | --num-futures) num_futures=${val} ;;
      --NUM-KEYWORDS | --num-keywords) num_keywords=${val} ;;
      --NUM-GENERATIONS | --num-generations) num_fixed_generations=${val} ;;
      --FIXED-LAMBDA | --fixed-lambda) fixed_lambda=${val} ;;
      --CONFIG-FILE | --config-file) config_file=${val} ;;
      --AGENT-MODEL-PATH | --agent-model-path) agent_model_path=${val} ;;
      --KEYWORDS-FILE-PATH | --keywords-file-path) keywords_file_path=${val} ;;
      --DATASTORE-FOR-SIMULATION | --datastore-for-simulation) datastore_for_simulation=${val} ;;
      --DATA-DIR | --preprocess_data-dir | --data-dir)
        data_dir=${val}
      ;;
      --LIMIT-TURNS-TO | --limit-turns-to) limit_turns_to=${val} ;;
      --SPLIT | --split) split=${val} ;;
      --ONE-STEP-GEN | --one-step-gen) one_step_gen=true ;;
      --NUM-SAVE-STEPS | --num-save-steps) num_intermediate_save_steps=${val} ;;
      --SIMULATION-DATASTORE | --simulation-datastore) simulation_datastore=${val} ;;
      --PARALLELIZE | --parallelize) run_parallel=true ;;
      -h | --help)
        display_help
        exit 0
        ;;
      *)
    esac
done

# Validate that run_approaches are provided
if [ "${#run_approaches[*]}" -eq 0 ]; then
    echo "--RUN-APPROACHES not provided"
    exit 1
fi

# Validate the model_path is provided if the model_type is finetuned
if [ "${model_type}" = "finetuned" ]; then
  if [ -z "${model_path}" ] && [ "${model_path}" = "" ]; then
    echo "If MODEL-TYPE is 'finetuned' --MODEL-PATH argument should be provided"
    exit 1
  fi
fi

# Validate preprocess_data dir is given
if [ -z "${data_dir}" ] || [ "${data_dir}" = "" ]; then
  echo "Please provide the --DATA-DIR"
  exit 1
fi

# Validate that the config file is provided
if [ -z "${config_file}" ] || [ "${config_file}" = "" ]; then
  echo "Please provide the --CONFIG-FILE"
  exit 1
fi

if [ eval_type == "simulated" ] && [ -z "${simulation_datastore}" ]; then
  echo "--simulated-datastore arg must be provided for eval-type 'simulated'"
  exit 1
fi

declare -A model_type_to_path=( ["finetuned"]="${model_path}" ["pretrained"]="gpt2-medium")

# display the approaches that are running
for approach in "${run_approaches[@]}";
do
  echo "${approach}"
done

echo ${model_type_to_path[$model_type]}

start=0
step=$((num_total_examples/num_intermediate_save_steps))
step=${step%.*}
echo "START=${start}, STEP=${step}, NUM_TOTAL_EXAMPLES=${num_total_examples}"
for num_eval_examples in $(eval echo "{${start}..${num_total_examples}..${step}}"); do  # Checking the intermediate save points for the experiment
    echo "${num_eval_examples}"
done

# Run all approaches
# for condition in ${args_to_conditions[$run_approaches]}; do
if [ -f "run_individual_progress.log" ]; then
  echo "Removing a previous progress log"
  rm "run_individual_progress.log"
fi

if [ "${eval_type}" == "real" ]; then
  all_keywords=(9)
else
  all_keywords=(9 7 5 3 1)
fi
# for num_keywords in {9,7,5,3,1}; do

proc_id=0
for num_eval_examples in $(eval echo "{${start}..${num_total_examples}..${step}}"); do  # This has been added to save intermediate results for smaller numbers of examples
    echo "NUM_EVAL_EXAMPLES=${num_eval_examples}"
    for condition in "${run_approaches[@]}"; do
        echo ${condition}
        for num_generations in '10'; do
            # for lambda in '15'; do
            for num_keywords in "${all_keywords[@]}"; do
                command="python run_approaches.py \
                        --save_dir ${save_dir} \
                        --model_path ${model_type_to_path[$model_type]} \
                        --eval_type ${eval_type} \
                        --config-file ${config_file} \
                        --condition ${condition} \
                        --data_dir ${data_dir} \
                        --num_eval_examples ${num_eval_examples} \
                        --num_total_examples ${num_total_examples} \
                        --num_keywords ${num_keywords} \
                        --num_futures ${num_futures} \
                        --split ${split} \
                        --fp16 "

                if [ ! -z "${one_step_gen}" ]; then
                    command+="--one-step-generation "
                fi

                # Only use GPU for CGMH approach
#               echo "approach ${condition}"
#               if [ "${condition}" != "CGMH" ]; then
#                   command+="--no-cuda "
#               fi
                echo "${simulation_datastore}"
                if [ ! -z "${simulation_datastore}" ]; then
                    command+="--datastore_for_simulation ${simulation_datastore} "
                fi

                if [ ! -z "${agent_model_path}" ]; then
                    command+="--agent_model_path ${agent_model_path} "
                fi

                if [ ! -z "${keywords_file_path}" ]; then
                    command+="--keywords_file_path ${keywords_file_path} "
                fi

                if [ ! -z "${limit_turns_to}" ]; then
                    command+="--limit-turns-to ${limit_turns_to} "
                fi

                if [ "${condition}" = "FuturesOfThePast" ]; then
                    command+="--lambda_param ${fixed_lambda} --num_candidate_generations ${num_generations}"
                elif [ "${condition}" = "DirectedBeamSearch" ]; then
                    command+="--lambda_param ${fixed_lambda}"
                elif [ "${condition}" = "RetrievalDBS" ]; then
                    command+="--lambda_param ${fixed_lambda}"
                elif [ "${condition}" = "CompressedRetrievalDBS" ]; then
                    command+="--lambda_param ${fixed_lambda} --to_compress"
                elif [ "${condition}" = "CompressedRetrieval" ]; then
                    command+="--to_compress"
                elif [ "${condition}" = "PromptingWithTraining" ]; then
                    command+="--prompting_model_path ${prompting_model_path}"
                elif [ "${condition}" = "WindowFuturesOfThePast" ]; then
                    command+="--lambda_param ${fixed_lambda} --num_candidate_generations ${num_generations}"
                elif [ "${condition}" = "WindowControlFuturesOfThePast" ]; then
                    command+="--lambda_param ${fixed_lambda} --num_candidate_generations ${num_generations}"
                fi

                echo "${command}"
                if [ -z "${run_parallel}" ]; then
                    eval "${command}"
                else
                    eval "${command} 2>&1 &"
                fi
            done
        done
    done
done

# Get plots
# python plotting/plot.py --save_dir ${save_dir}
