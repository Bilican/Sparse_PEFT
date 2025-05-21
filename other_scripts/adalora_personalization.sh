# --- Configuration --- #
ALPHAS=(32)
# Define fixed AdaLoRA parameters for this search (Adjust as needed based on your base config)
RANK=1 # Example: Read from your original adalora_config.yaml or set desired fixed rank
TARGET_MODULES='["to_q", "to_k", "to_v", "to_out.0"]' # Example: Common target modules
INIT_R=2 # Example: AdaLoRA specific param
# Base directory for all outputs (Updated for AdaLoRA alpha search)
BASE_OUTPUT_DIR="./outputs_adalora_alpha_search"
# Use the default base training config
TRAINING_CONFIG_PATH="./sdxl_dreambooth/config/training_config.yaml"
# Use the default experiment config for evaluation
EVALUATION_CONFIG_PATH="./sdxl_dreambooth/config/experiment_config.yaml"

# --- Derived Paths --- #
PROJECT_DIR=$(pwd) # Capture project dir early

# Set Hugging Face cache directory to be local to the project
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

echo "======================================="
echo "Starting AdaLoRA Alpha Hyperparameter Search"
echo "Fixed Rank: ${RANK}" # Assuming fixed rank for alpha search
echo "Alphas to test: ${ALPHAS[@]}"
echo "Base Output Dir: ${PROJECT_DIR}/${BASE_OUTPUT_DIR}"
echo "HF Cache Dir: ${HF_HOME}"
echo "======================================="


# Redefine PROJECT_DIR after cd
PROJECT_DIR=$(pwd)
echo "Current working directory: ${PROJECT_DIR}"


# Define object instances and prompts file
OBJECT_INSTANCES=(
    "backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle"
    "clock" "colorful_sneaker" "duck_toy" "fancy_boot" "grey_sloth_plushie"
    "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon"
    "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie"
)
OBJECT_PROMPTS_FILE="${PROJECT_DIR}/dataset/prompts_and_classes_obj.txt" # Use absolute/relative from PROJECT_DIR

# Define live instances and prompts file
LIVE_INSTANCES=(
    "cat" "cat2" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8"
)
LIVE_PROMPTS_FILE="${PROJECT_DIR}/dataset/prompts_and_classes_live.txt" # Use absolute/relative from PROJECT_DIR
# Example: Limit instances for faster testing (optional)

# Set the unique token for all training and evaluation
UNIQUE_TOKEN="zjw" # Make sure this matches the token in experiment_config.yaml

# Function to get the class name for an instance from a prompt file
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

# Common evaluation parameters
MIXED_PRECISION="bf16"
NUM_IMAGES_PER_PROMPT=4
MAX_PROMPTS=25 # Adjust if needed
SEED=42
GUIDANCE_SCALE=7.5
INFERENCE_STEPS=30
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0" # Use model ID to leverage HF_HOME cache
USE_UNIQUE_SEEDS=true

# --- Main Loop for Alphas --- #
for ALPHA in "${ALPHAS[@]}"; do
    ADAPTER_TYPE="adalora"
    # Construct config name and temporary config file path
    CONFIG_NAME="${ADAPTER_TYPE}_rank_${RANK}_alpha_${ALPHA}" # Include rank in name for clarity
    TEMP_ADAPTER_CONFIG_PATH="./temp_adapter_config_${CONFIG_NAME}.yaml" # Relative to PROJECT_DIR

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
alpha: ${ALPHA}
target_modules: ${TARGET_MODULES}
init_r: ${INIT_R}
tinit: ${TINIT}
tfinal: ${TFINAL}
deltaT: ${DELTAT}
# Add other necessary AdaLoRA params like beta1, beta2, orth_reg_weight if needed
# beta1: 0.9
# beta2: 0.999
# orth_reg_weight: 0.1
# bias: "none" # Example
# module_dropout: 0.0 # Example
EOF

    if [ ! -f "$TEMP_ADAPTER_CONFIG_PATH" ]; then
        echo "Error: Failed to create temporary adapter config file: ${TEMP_ADAPTER_CONFIG_PATH}" >&2
        continue # Skip to the next alpha
    fi
    echo "Temporary adapter config created: ${TEMP_ADAPTER_CONFIG_PATH}"

    # Construct the specific output directory for this training run (relative to project dir)
    TRAINING_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    # Construct the absolute path for checking existence and passing to python
    ABS_TRAINING_OUTPUT_DIR="${PROJECT_DIR}/${TRAINING_OUTPUT_DIR}"

    echo "======================================="
    echo "Configuration for Rank ${RANK}, Alpha ${ALPHA}"
    echo "Adapter Config (Temp): ${TEMP_ADAPTER_CONFIG_PATH}"
    echo "Training Config: ${TRAINING_CONFIG_PATH}" # Using default training config
    echo "Evaluation Config: ${EVALUATION_CONFIG_PATH}" # Using default eval config
    echo "Output Dir:     ${ABS_TRAINING_OUTPUT_DIR}"
    echo "======================================="

    # --- Process Object Instances for current Rank/Alpha --- #
    echo "======================================="
    echo "Processing Object Instances (Rank ${RANK}, Alpha ${ALPHA})"
    echo "======================================="

    for INSTANCE in "${OBJECT_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})"
        echo "======================================="

        # Get the correct class name for this instance
        CLASS_NAME=$(get_class_name "$INSTANCE" "$OBJECT_PROMPTS_FILE")
        echo "Instance: ${INSTANCE}, Class: ${CLASS_NAME}"

        # Set instance-specific paths
        INSTANCE_OUTPUT_DIR="${ABS_TRAINING_OUTPUT_DIR}/${INSTANCE}"
        INSTANCE_DATA_DIR="./dataset/${INSTANCE}" # Relative to PROJECT_DIR
        INSTANCE_MODEL_DIR="${INSTANCE_OUTPUT_DIR}/final_model"

        # Check if the final model directory already exists for this instance
        if [ -d "${INSTANCE_MODEL_DIR}" ]; then
            echo "Final model directory already exists: ${INSTANCE_MODEL_DIR}"
            echo "Skipping training for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})."
        else
            echo "Starting AdaLoRA Training Run for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})..."
            # Ensure the instance output directory exists
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            # Run training with instance data directory
            # Pass training_config, *temporary* adapter_config, and instance specifics
            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TRAINING_CONFIG_PATH" \
                --adapter_config "$TEMP_ADAPTER_CONFIG_PATH" \
                --experiment_config "$EVALUATION_CONFIG_PATH" \
                --output_dir "$INSTANCE_OUTPUT_DIR" \
                --instance_data_dir "$INSTANCE_DATA_DIR" \
                --instance_prompt "a ${UNIQUE_TOKEN} ${CLASS_NAME}" \
                --class_name "${CLASS_NAME}"

            # Check if training produced the final model directory
            if [ ! -d "${INSTANCE_MODEL_DIR}" ]; then
                echo "Error: Training finished but final model directory was not created: ${INSTANCE_MODEL_DIR}" >&2
                # Continue to next instance instead of exiting
                continue
            fi
            echo "Training for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA}) finished successfully."
        fi

        # --- Evaluation --- #
        # Check if evaluation results already exist for this instance
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")

        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            # If results found, skip evaluation
            echo "======================================="
            echo "Skipping Evaluation for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})"
            echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
            echo "======================================="
        else
            # If no results found, proceed with evaluation
            echo "======================================="
            echo "Starting Evaluation for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})"
            echo "======================================="

            # Define the specific adapter directory from the training output (absolute path)
            ADAPTER_DIR="${INSTANCE_MODEL_DIR}"

            # Define the output directory for this evaluation run, inside the instance output dir (absolute path)
            EVALUATION_OUTPUT_DIR="${INSTANCE_OUTPUT_DIR}/evaluation_results_$(date +%Y%m%d_%H%M%S)"

            # Ensure evaluation output directory exists
            mkdir -p "$EVALUATION_OUTPUT_DIR"

            echo "Using adapter from: ${ADAPTER_DIR}"
            echo "Saving evaluation results to: ${EVALUATION_OUTPUT_DIR}"

            # Check if adapter directory exists before running evaluation
            if [ ! -d "$ADAPTER_DIR" ]; then
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})." >&2
                # Continue to next instance instead of exiting
                continue
            fi

            # Run evaluation using the trained adapter
            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EVALUATION_CONFIG_PATH" \
              --output_dir "$EVALUATION_OUTPUT_DIR" \
              --prompts_file "$OBJECT_PROMPTS_FILE" \
              --adapter_dirs "$ADAPTER_DIR" \
              --concepts "$INSTANCE" \
              --device cuda \
              --mixed_precision "$MIXED_PRECISION" \
              --pretrained_model_path "$PRETRAINED_MODEL" \
              --num_images "$NUM_IMAGES_PER_PROMPT" \
              --max_prompts "$MAX_PROMPTS" \
              ${USE_UNIQUE_SEEDS:+--use_unique_seeds} \
              --seed "$SEED" \
              --num_inference_steps "$INFERENCE_STEPS" \
              --guidance_scale "$GUIDANCE_SCALE" \
              --unique_token "$UNIQUE_TOKEN"

            # Check if evaluation actually produced images
            if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
                echo "Error: Evaluation finished but no image files (.png/.jpg) were found in ${EVALUATION_OUTPUT_DIR}" >&2
            else
                echo "Evaluation complete for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})! Results saved to ${EVALUATION_OUTPUT_DIR}"
            fi
        fi
    done

    # --- Process Live Instances for current Rank/Alpha --- #
    echo ""
    echo "======================================="
    echo "Processing Live Instances (Rank ${RANK}, Alpha ${ALPHA})"
    echo "======================================="

    for INSTANCE in "${LIVE_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})"
        echo "======================================="

        # Get the correct class name for this instance
        CLASS_NAME=$(get_class_name "$INSTANCE" "$LIVE_PROMPTS_FILE")
        echo "Instance: ${INSTANCE}, Class: ${CLASS_NAME}"

        # Set instance-specific paths
        INSTANCE_OUTPUT_DIR="${ABS_TRAINING_OUTPUT_DIR}/${INSTANCE}"
        INSTANCE_DATA_DIR="./dataset/${INSTANCE}" # Relative to PROJECT_DIR
        INSTANCE_MODEL_DIR="${INSTANCE_OUTPUT_DIR}/final_model"

        # Check if the final model directory already exists for this instance
        if [ -d "${INSTANCE_MODEL_DIR}" ]; then
            echo "Final model directory already exists: ${INSTANCE_MODEL_DIR}"
            echo "Skipping training for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})."
        else
            echo "Starting AdaLoRA Training Run for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})..."
            # Ensure the instance output directory exists
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            # Run training with instance data directory
            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TRAINING_CONFIG_PATH" \
                --adapter_config "$TEMP_ADAPTER_CONFIG_PATH" \
                --experiment_config "$EVALUATION_CONFIG_PATH" \
                --output_dir "$INSTANCE_OUTPUT_DIR" \
                --instance_data_dir "$INSTANCE_DATA_DIR" \
                --instance_prompt "a ${UNIQUE_TOKEN} ${CLASS_NAME}" \
                --class_name "${CLASS_NAME}"

            # Check if training produced the final model directory
            if [ ! -d "${INSTANCE_MODEL_DIR}" ]; then
                echo "Error: Training finished but final model directory was not created: ${INSTANCE_MODEL_DIR}" >&2
                # Continue to next instance instead of exiting
                continue
            fi
            echo "Training for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA}) finished successfully."
        fi

        # --- Evaluation --- #
        # Check if evaluation results already exist for this instance
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")

        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            # If results found, skip evaluation
            echo "======================================="
            echo "Skipping Evaluation for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})"
            echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
            echo "======================================="
        else
            # If no results found, proceed with evaluation
            echo "======================================="
            echo "Starting Evaluation for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})"
            echo "======================================="

            # Define the specific adapter directory from the training output (absolute path)
            ADAPTER_DIR="${INSTANCE_MODEL_DIR}"

            # Define the output directory for this evaluation run, inside the instance output dir (absolute path)
            EVALUATION_OUTPUT_DIR="${INSTANCE_OUTPUT_DIR}/evaluation_results_$(date +%Y%m%d_%H%M%S)"

            # Ensure evaluation output directory exists
            mkdir -p "$EVALUATION_OUTPUT_DIR"

            echo "Using adapter from: ${ADAPTER_DIR}"
            echo "Saving evaluation results to: ${EVALUATION_OUTPUT_DIR}"

            # Check if adapter directory exists before running evaluation
            if [ ! -d "$ADAPTER_DIR" ]; then
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})." >&2
                # Continue to next instance instead of exiting
                continue
            fi

            # Run evaluation using the trained adapter with LIVE prompts file
            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EVALUATION_CONFIG_PATH" \
              --output_dir "$EVALUATION_OUTPUT_DIR" \
              --prompts_file "$LIVE_PROMPTS_FILE" \
              --adapter_dirs "$ADAPTER_DIR" \
              --concepts "$INSTANCE" \
              --device cuda \
              --mixed_precision "$MIXED_PRECISION" \
              --pretrained_model_path "$PRETRAINED_MODEL" \
              --num_images "$NUM_IMAGES_PER_PROMPT" \
              --max_prompts "$MAX_PROMPTS" \
              ${USE_UNIQUE_SEEDS:+--use_unique_seeds} \
              --seed "$SEED" \
              --num_inference_steps "$INFERENCE_STEPS" \
              --guidance_scale "$GUIDANCE_SCALE" \
              --unique_token "$UNIQUE_TOKEN"

            # Check if evaluation actually produced images
            if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
                echo "Error: Evaluation finished but no image files (.png/.jpg) were found in ${EVALUATION_OUTPUT_DIR}" >&2
            else
                echo "Evaluation complete for ${INSTANCE} (Rank ${RANK}, Alpha ${ALPHA})! Results saved to ${EVALUATION_OUTPUT_DIR}"
            fi
        fi
    done

    # --- Clean up temporary config file --- #
    echo "Removing temporary adapter config file: ${TEMP_ADAPTER_CONFIG_PATH}"
    rm -f "$TEMP_ADAPTER_CONFIG_PATH"

done # End of loop for ALPHAS

echo ""
echo "======================================="
echo "All AdaLoRA alpha values processed successfully (Rank ${RANK})" # Updated text
echo "Outputs saved in ${PROJECT_DIR}/${BASE_OUTPUT_DIR}"
echo "=======================================" 