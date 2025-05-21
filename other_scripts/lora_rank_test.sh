# --- Configuration --- #
# Define ranks to test
#RANKS=(1 2 4)
RANKS=(1 2 3 4)
# Instance and Prompt Definitions (Names only)
# OBJECT_INSTANCES=("teapot" "can" "duck_toy" "robot_toy")
OBJECT_INSTANCES=(
    "backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle"
    "clock" "colorful_sneaker" "duck_toy" "fancy_boot" "grey_sloth_plushie"
    "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon"
    "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie"
)
OBJECT_PROMPT_FILENAME="prompts_and_classes_obj.txt"
# LIVE_INSTANCES=("cat" "dog" "dog6" "dog2")
LIVE_INSTANCES=(
    "cat" "cat2" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8"
)
LIVE_PROMPT_FILENAME="prompts_and_classes_live.txt"
# Unique Token (Can be overridden by experiment config)
UNIQUE_TOKEN="zjw"
# Target Project Directory
PEFT_LEAGUE_DIR="."

# --- Common Evaluation Parameters --- #
# These are mostly static or passed to python scripts
NUM_IMAGES_PER_PROMPT=4
MAX_PROMPTS=25
SEED=42
GUIDANCE_SCALE=7.5
INFERENCE_STEPS=30
MIXED_PRECISION="bf16"

# --- Main Setup --- #
echo "======================================="
echo "Starting LoRA Rank Evaluation"
echo "Target Project Dir: ${PEFT_LEAGUE_DIR}"
echo "Ranks to test: ${RANKS[@]}"
echo "======================================="



# --- Define Paths AFTER CD --- #
PROJECT_DIR=$(pwd) 
echo "Current working directory: ${PROJECT_DIR}"

# Base directory for all outputs (relative to PROJECT_DIR)
BASE_OUTPUT_DIR="./outputs_lora_gpu_test"
# Use a single base training config (relative to PROJECT_DIR)
TRAINING_CONFIG_PATH="./sdxl_dreambooth/config/training_config.yaml"
# Experiment config (relative to PROJECT_DIR)
EXPERIMENT_CONFIG_PATH="./sdxl_dreambooth/config/experiment_config.yaml"
# Prompt files (relative to PROJECT_DIR)
OBJECT_PROMPTS_FILE="/home/abilican21/Peft-League/dataset/${OBJECT_PROMPT_FILENAME}"
LIVE_PROMPTS_FILE="/home/abilican21/Peft-League/dataset/${LIVE_PROMPT_FILENAME}"
# Default LoRA target modules
LORA_TARGET_MODULES='["to_q", "to_k", "to_v", "to_out.0"]'
# Pretrained Model Path (relative to PROJECT_DIR)
PRETRAINED_MODEL="/home/abilican21/Peft-League/pretrained_sdxl_model/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b" # Use absolute path for robustness

# --- Function to get class name --- #
get_class_name() {
    local instance=$1
    local prompts_file=$2
    # Extract the class from the prompts file based on the instance name
    local class=$(grep "^$instance," "$prompts_file" | cut -d',' -f2)
    # If class is empty, use the instance name itself
    if [ -z "$class" ]; then
        class="$instance"
    fi
    echo "$class"
}

# --- Main Loop for Ranks --- #
for RANK in "${RANKS[@]}"; do
    ALPHA=$RANK # Set alpha equal to rank
    ADAPTER_TYPE="lora"
    CONFIG_NAME="${ADAPTER_TYPE}_rank_${RANK}_alpha_${ALPHA}"
    # Place temp config in current dir (PROJECT_DIR)
    TEMP_ADAPTER_CONFIG_PATH="./temp_adapter_config_${CONFIG_NAME}.yaml"

    echo ""
    echo "***************************************"
    echo "Processing Rank: ${RANK}, Alpha: ${ALPHA}"
    echo "Config Name:     ${CONFIG_NAME}"
    echo "Temp Config File: ${TEMP_ADAPTER_CONFIG_PATH}"
    echo "***************************************"

    # --- Generate Temporary Adapter Config --- #
    echo "Generating temporary adapter config..."
    cat << EOF > "$TEMP_ADAPTER_CONFIG_PATH"
adapter_type: ${ADAPTER_TYPE}
rank: ${RANK}
alpha: ${ALPHA} # Same as lora_alpha for LoRA
target_modules: ${LORA_TARGET_MODULES}
module_dropout: 0.0
bias: "none"
# learning_rate and max_train_steps should be defined in TRAINING_CONFIG_PATH
# or overridden if needed per rank, but keeping base config for now.
EOF

    if [ ! -f "$TEMP_ADAPTER_CONFIG_PATH" ]; then
        echo "Error: Failed to create temporary adapter config file: ${TEMP_ADAPTER_CONFIG_PATH}" >&2
        continue # Skip to the next rank
    fi
    echo "Temporary adapter config created."

    # Construct the specific output directory for this training run (relative to PROJECT_DIR)
    TRAINING_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    # Construct the absolute path for checking existence and passing to python
    # PROJECT_DIR is now correctly set after cd
    ABS_TRAINING_OUTPUT_DIR="${PROJECT_DIR}/${TRAINING_OUTPUT_DIR}"

    echo "======================================="
    echo "Configuration for Rank ${RANK}"
    echo "Adapter Config: ${TEMP_ADAPTER_CONFIG_PATH}"
    echo "Base Training Config: ${TRAINING_CONFIG_PATH}"
    echo "Experiment Config: ${EXPERIMENT_CONFIG_PATH}"
    echo "Output Dir:     ${ABS_TRAINING_OUTPUT_DIR}"
    echo "======================================="

    # --- Process Object Instances for current Rank --- #
    echo "======================================="
    echo "Processing Object Instances for Rank ${RANK}"
    echo "======================================="

    for INSTANCE in "${OBJECT_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (Rank ${RANK})"
        echo "======================================="

        CLASS_NAME=$(get_class_name "$INSTANCE" "$OBJECT_PROMPTS_FILE")
        echo "Instance: ${INSTANCE}, Class: ${CLASS_NAME}"

        INSTANCE_OUTPUT_DIR="${ABS_TRAINING_OUTPUT_DIR}/${INSTANCE}"
        INSTANCE_DATA_DIR="./dataset/${INSTANCE}" # Relative to PROJECT_DIR
        INSTANCE_MODEL_DIR="${INSTANCE_OUTPUT_DIR}/final_model"

        if [ -d "${INSTANCE_MODEL_DIR}" ]; then
            echo "Final model directory already exists: ${INSTANCE_MODEL_DIR}"
            echo "Skipping training for ${INSTANCE} (Rank ${RANK})."
        else
            echo "Starting Training Run for ${INSTANCE} (Rank ${RANK})..."
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TRAINING_CONFIG_PATH" \
                --adapter_config "$TEMP_ADAPTER_CONFIG_PATH" \
                --experiment_config "$EXPERIMENT_CONFIG_PATH" \
                --output_dir "$INSTANCE_OUTPUT_DIR" \
                --instance_data_dir "$INSTANCE_DATA_DIR" \
                --instance_prompt "a ${UNIQUE_TOKEN} ${CLASS_NAME}" \
                --class_name "${CLASS_NAME}"

            if [ ! -d "${INSTANCE_MODEL_DIR}" ]; then
                echo "Error: Training finished but final model directory was not created: ${INSTANCE_MODEL_DIR}" >&2
                continue # Continue to next instance
            fi
            echo "Training for ${INSTANCE} (Rank ${RANK}) finished successfully."
        fi

        # --- Evaluation for Object Instance --- #
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")
        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            echo "Skipping Evaluation for ${INSTANCE} (Rank ${RANK}). Found results."
        else
            echo "Starting Evaluation for ${INSTANCE} (Rank ${RANK})..."
            ADAPTER_DIR="${INSTANCE_MODEL_DIR}" # Use the trained model dir
            EVALUATION_OUTPUT_DIR="${INSTANCE_OUTPUT_DIR}/evaluation_results_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$EVALUATION_OUTPUT_DIR"

            echo "Using adapter from: ${ADAPTER_DIR}"
            echo "Saving evaluation results to: ${EVALUATION_OUTPUT_DIR}"

            if [ ! -d "$ADAPTER_DIR" ]; then
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (Rank ${RANK})." >&2
                continue # Continue to next instance
            fi

            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EXPERIMENT_CONFIG_PATH" \
              --pretrained_model_path "$PRETRAINED_MODEL" \
              --output_dir "$EVALUATION_OUTPUT_DIR" \
              --prompts_file "$OBJECT_PROMPTS_FILE" \
              --adapter_dirs "$ADAPTER_DIR" \
              --num_images "$NUM_IMAGES_PER_PROMPT" \
              --max_prompts "$MAX_PROMPTS" \
              --use_unique_seeds \
              --concepts "$INSTANCE" \
              --seed "$SEED" \
              --device cuda \
              --mixed_precision "$MIXED_PRECISION" \
              --num_inference_steps "$INFERENCE_STEPS" \
              --guidance_scale "$GUIDANCE_SCALE" \
              --unique_token "$UNIQUE_TOKEN"

            if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
                echo "Error: Evaluation finished but no image files found in ${EVALUATION_OUTPUT_DIR}" >&2
            else
                echo "Evaluation complete for ${INSTANCE} (Rank ${RANK})!"
            fi
        fi
    done

    # --- Process Live Instances for current Rank --- #
    echo ""
    echo "======================================="
    echo "Processing Live Instances for Rank ${RANK}"
    echo "======================================="

    for INSTANCE in "${LIVE_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (Rank ${RANK})"
        echo "======================================="

        CLASS_NAME=$(get_class_name "$INSTANCE" "$LIVE_PROMPTS_FILE")
        echo "Instance: ${INSTANCE}, Class: ${CLASS_NAME}"

        INSTANCE_OUTPUT_DIR="${ABS_TRAINING_OUTPUT_DIR}/${INSTANCE}"
        INSTANCE_DATA_DIR="./dataset/${INSTANCE}" # Relative to PROJECT_DIR
        INSTANCE_MODEL_DIR="${INSTANCE_OUTPUT_DIR}/final_model"

        if [ -d "${INSTANCE_MODEL_DIR}" ]; then
            echo "Final model directory already exists: ${INSTANCE_MODEL_DIR}"
            echo "Skipping training for ${INSTANCE} (Rank ${RANK})."
        else
            echo "Starting Training Run for ${INSTANCE} (Rank ${RANK})..."
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TRAINING_CONFIG_PATH" \
                --adapter_config "$TEMP_ADAPTER_CONFIG_PATH" \
                --experiment_config "$EXPERIMENT_CONFIG_PATH" \
                --output_dir "$INSTANCE_OUTPUT_DIR" \
                --instance_data_dir "$INSTANCE_DATA_DIR" \
                --instance_prompt "a ${UNIQUE_TOKEN} ${CLASS_NAME}" \
                --class_name "${CLASS_NAME}"

            if [ ! -d "${INSTANCE_MODEL_DIR}" ]; then
                echo "Error: Training finished but final model directory was not created: ${INSTANCE_MODEL_DIR}" >&2
                continue # Continue to next instance
            fi
            echo "Training for ${INSTANCE} (Rank ${RANK}) finished successfully."
        fi

        # --- Evaluation for Live Instance --- #
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")
        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            echo "Skipping Evaluation for ${INSTANCE} (Rank ${RANK}). Found results."
        else
            echo "Starting Evaluation for ${INSTANCE} (Rank ${RANK})..."
            ADAPTER_DIR="${INSTANCE_MODEL_DIR}" # Use the trained model dir
            EVALUATION_OUTPUT_DIR="${INSTANCE_OUTPUT_DIR}/evaluation_results_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$EVALUATION_OUTPUT_DIR"

            echo "Using adapter from: ${ADAPTER_DIR}"
            echo "Saving evaluation results to: ${EVALUATION_OUTPUT_DIR}"

            if [ ! -d "$ADAPTER_DIR" ]; then
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (Rank ${RANK})." >&2
                continue # Continue to next instance
            fi

            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EXPERIMENT_CONFIG_PATH" \
              --pretrained_model_path "$PRETRAINED_MODEL" \
              --output_dir "$EVALUATION_OUTPUT_DIR" \
              --prompts_file "$LIVE_PROMPTS_FILE" \
              --adapter_dirs "$ADAPTER_DIR" \
              --num_images "$NUM_IMAGES_PER_PROMPT" \
              --max_prompts "$MAX_PROMPTS" \
              --use_unique_seeds \
              --concepts "$INSTANCE" \
              --seed "$SEED" \
              --device cuda \
              --mixed_precision "$MIXED_PRECISION" \
              --num_inference_steps "$INFERENCE_STEPS" \
              --guidance_scale "$GUIDANCE_SCALE" \
              --unique_token "$UNIQUE_TOKEN"

            if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
                echo "Error: Evaluation finished but no image files found in ${EVALUATION_OUTPUT_DIR}" >&2
            else
                echo "Evaluation complete for ${INSTANCE} (Rank ${RANK})!"
            fi
        fi
    done

    # --- Clean up temporary config file --- #
    echo "Removing temporary adapter config file: ${TEMP_ADAPTER_CONFIG_PATH}"
    rm -f "$TEMP_ADAPTER_CONFIG_PATH"

done # End of loop for RANKS

echo ""
echo "======================================="
echo "All LoRA ranks processed successfully"
echo "Outputs saved in ${PROJECT_DIR}/${BASE_OUTPUT_DIR}" # PROJECT_DIR is now correct
echo "=======================================" 