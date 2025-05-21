# --- Configuration --- #
# Define the FourierFT scaling values to test
SCALES=(64) # Example range, adjust as needed
# Base directory for all outputs (Updated for FourierFT scale search)
BASE_OUTPUT_DIR="./outputs_fourierft_scale_search" # Updated path

# Use the specified FP32 training config
TRAINING_CONFIG_PATH="./sdxl_dreambooth/config/training_configfp32.yaml"
# Path to the main experiment configuration for evaluation
EXPERIMENT_CONFIG_PATH="./sdxl_dreambooth/config/experiment_fourierft_config.yaml" # Keep using this for eval params if it defines them

# --- Base FourierFT Config Parameters (from fourierft_config.yaml) ---
# These will be used to generate temporary configs
ADAPTER_TYPE="fourierft"
TARGET_MODULES='["to_q", "to_k", "to_v", "to_out.0"]'
N_FREQUENCY=2592
# MODULES_TO_SAVE=null # Assuming null is default

# --- Derived Paths --- #
PROJECT_DIR=$(pwd) # Capture project dir early

# Set Hugging Face cache directory to be local to the project
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

echo "======================================="
echo "Starting FourierFT Scaling Hyperparameter Search" # Updated text
echo "Scales to test: ${SCALES[@]}"
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
# Uncomment below to test with a single instance
#OBJECT_INSTANCES=()
#LIVE_INSTANCES=("dog")

# Set the unique token for all training and evaluation
UNIQUE_TOKEN="zjw" # Make sure this matches the token in experiment_fourierft_config.yaml

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

# Common evaluation parameters (Explicitly defined like LoHa/LoKR scripts)
MIXED_PRECISION="no"
NUM_IMAGES_PER_PROMPT=4
MAX_PROMPTS=25 # Increased max prompts for FourierFT eval
SEED=42
GUIDANCE_SCALE=7.5
INFERENCE_STEPS=30
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0" # Use model ID to leverage HF_HOME cache
USE_UNIQUE_SEEDS=true # Add this for consistency if desired

# --- Main Loop for Scales --- #
for SCALE in "${SCALES[@]}"; do
    # Construct config name and temporary config file path
    CONFIG_NAME="${ADAPTER_TYPE}_scale_${SCALE}"
    TEMP_ADAPTER_CONFIG_PATH="./temp_adapter_config_${CONFIG_NAME}.yaml" # Relative to PROJECT_DIR

    echo ""
    echo "***************************************"
    echo "Processing Scale: ${SCALE}"
    echo "Config Name:     ${CONFIG_NAME}"
    echo "Temp Config File: ${TEMP_ADAPTER_CONFIG_PATH}"
    echo "***************************************"

    # --- Generate Temporary Adapter Config --- #
    echo "Generating temporary adapter config..."
    cat << EOF > "$TEMP_ADAPTER_CONFIG_PATH"
adapter_type: ${ADAPTER_TYPE}
target_modules: ${TARGET_MODULES}
modules_to_save: null # Or omit if null is default
# FourierFT specific parameters
n_frequency: ${N_FREQUENCY}
scaling: ${SCALE} # Use the current SCALE from the loop
random_loc_seed: ${RANDOM_LOC_SEED}
fan_in_fan_out: ${FAN_IN_FAN_OUT}
bias: "${BIAS}"
init_weights: ${INIT_WEIGHTS}
EOF

    if [ ! -f "$TEMP_ADAPTER_CONFIG_PATH" ]; then
        echo "Error: Failed to create temporary adapter config file: ${TEMP_ADAPTER_CONFIG_PATH}" >&2
        continue # Skip to the next scale
    fi
    echo "Temporary adapter config created: ${TEMP_ADAPTER_CONFIG_PATH}"

    # Construct the specific output directory for this training run (relative to project dir)
    TRAINING_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    # Construct the absolute path for checking existence and passing to python
    ABS_TRAINING_OUTPUT_DIR="${PROJECT_DIR}/${TRAINING_OUTPUT_DIR}"

    echo "======================================="
    echo "Configuration for Scale ${SCALE}"
    echo "Adapter Config (Temp): ${TEMP_ADAPTER_CONFIG_PATH}"
    echo "Training Config (FP32): ${TRAINING_CONFIG_PATH}" # Using specified FP32 training config
    echo "Experiment Config (Eval): ${EXPERIMENT_CONFIG_PATH}" # Using experiment config for eval params
    echo "Output Dir:     ${ABS_TRAINING_OUTPUT_DIR}"
    echo "======================================="

    # --- Process Object Instances for current Scale --- #
    echo "======================================="
    echo "Processing Object Instances (Scale ${SCALE})"
    echo "======================================="

    for INSTANCE in "${OBJECT_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (Scale ${SCALE})"
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
            echo "Skipping training for ${INSTANCE} (Scale ${SCALE})."
        else
            echo "Starting FourierFT Training Run for ${INSTANCE} (Scale ${SCALE})..." # Updated text
            # Ensure the instance output directory exists
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            # Run training with instance data directory
            # Pass training_config, experiment_config, *temporary* adapter_config, and instance specifics
            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TRAINING_CONFIG_PATH" \
                --experiment_config "$EXPERIMENT_CONFIG_PATH" \
                --adapter_config "$TEMP_ADAPTER_CONFIG_PATH" \
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
            echo "Training for ${INSTANCE} (Scale ${SCALE}) finished successfully."
        fi

        # --- Evaluation --- #
        # Check if evaluation results already exist for this instance
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")

        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            # If results found, skip evaluation
            echo "======================================="
            echo "Skipping Evaluation for ${INSTANCE} (Scale ${SCALE})"
            echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
            echo "======================================="
        else
            # If no results found, proceed with evaluation
            echo "======================================="
            echo "Starting Evaluation for ${INSTANCE} (Scale ${SCALE})"
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
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (Scale ${SCALE})." >&2
                # Continue to next instance instead of exiting
                continue
            fi

            # Run evaluation using the trained adapter
            # Pass the experiment config (potentially overriding some params), specific adapter dir, instance info, and common eval params
            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EXPERIMENT_CONFIG_PATH" \
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
                echo "Evaluation complete for ${INSTANCE} (Scale ${SCALE})! Results saved to ${EVALUATION_OUTPUT_DIR}"
            fi
        fi
    done

    # --- Process Live Instances for current Scale --- #
    echo ""
    echo "======================================="
    echo "Processing Live Instances (Scale ${SCALE})"
    echo "======================================="

    for INSTANCE in "${LIVE_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (Scale ${SCALE})"
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
            echo "Skipping training for ${INSTANCE} (Scale ${SCALE})."
        else
            echo "Starting FourierFT Training Run for ${INSTANCE} (Scale ${SCALE})..." # Updated text
            # Ensure the instance output directory exists
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            # Run training with instance data directory
            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TRAINING_CONFIG_PATH" \
                --experiment_config "$EXPERIMENT_CONFIG_PATH" \
                --adapter_config "$TEMP_ADAPTER_CONFIG_PATH" \
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
            echo "Training for ${INSTANCE} (Scale ${SCALE}) finished successfully."
        fi

        # --- Evaluation --- #
        # Check if evaluation results already exist for this instance
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")

        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            # If results found, skip evaluation
            echo "======================================="
            echo "Skipping Evaluation for ${INSTANCE} (Scale ${SCALE})"
            echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
            echo "======================================="
        else
            # If no results found, proceed with evaluation
            echo "======================================="
            echo "Starting Evaluation for ${INSTANCE} (Scale ${SCALE})"
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
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (Scale ${SCALE})." >&2
                # Continue to next instance instead of exiting
                continue
            fi

            # Run evaluation using the trained adapter with LIVE prompts file
            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EXPERIMENT_CONFIG_PATH" \
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
                echo "Evaluation complete for ${INSTANCE} (Scale ${SCALE})! Results saved to ${EVALUATION_OUTPUT_DIR}"
            fi
        fi
    done

    # --- Clean up temporary config file --- #
    echo "Removing temporary adapter config file: ${TEMP_ADAPTER_CONFIG_PATH}"
    rm -f "$TEMP_ADAPTER_CONFIG_PATH"

done # End of loop for SCALES

echo ""
echo "======================================="
echo "All FourierFT scale values processed successfully" # Updated text
echo "Outputs saved in ${PROJECT_DIR}/${BASE_OUTPUT_DIR}"
echo "=======================================" 