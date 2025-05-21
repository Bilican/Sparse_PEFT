# --- Configuration --- #
# Define the specific learning rates to test numerically
#LEARNING_RATES=(0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128 0.0256)
LEARNING_RATES=(0.0032 0.0064)
#TEXT_ENCODER_LRS=(0.0001 0.0002 0.0004 0.0008 0.0016 0.0032 0.0064 0.0128)
# Define the fixed adapter config for this run (VeRA)
ADAPTER_CONFIG_PATH="./sdxl_dreambooth/config/vera_config.yaml" # Fixed VeRA config
# Base directory for all outputs (Updated for VeRA LR search)
BASE_OUTPUT_DIR="./outputs_vera_lr_search"
# Base training config template path
BASE_TRAINING_CONFIG_PATH="./sdxl_dreambooth/config/training_config_vera.yaml" # Use the VeRA specific base config
# Use the default experiment config for evaluation
EVALUATION_CONFIG_PATH="./sdxl_dreambooth/config/experiment_config.yaml"

# --- Derived Paths --- #
PROJECT_DIR=$(pwd) # Capture project dir early

echo "======================================="
echo "Starting VeRA Learning Rate Hyperparameter Search"
echo "Learning Rates to test: ${LEARNING_RATES[@]}"
echo "Text Encoder LRs to test: ${TEXT_ENCODER_LRS[@]}"
echo "Fixed Adapter Config: ${ADAPTER_CONFIG_PATH}"
echo "Base Training Config: ${PROJECT_DIR}/${BASE_TRAINING_CONFIG_PATH}"
echo "Base Output Dir: ${PROJECT_DIR}/${BASE_OUTPUT_DIR}"
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
# Common evaluation parameters (Many should come from EVALUATION_CONFIG_PATH)
MIXED_PRECISION="bf16" # Keep if not set in config
NUM_IMAGES_PER_PROMPT=4
MAX_PROMPTS=25 # Adjust if needed
SEED=42
GUIDANCE_SCALE=7.5
INFERENCE_STEPS=30
PRETRAINED_MODEL="${PROJECT_DIR}/pretrained_sdxl_model/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b" # Use absolute/relative from PROJECT_DIR (ensure this is correct)
USE_UNIQUE_SEEDS=true # Set evaluation unique seeds flag
# UNIQUE_TOKEN is already defined above

# --- Main Loop for LR Multipliers --- #
# Get the number of learning rates to test
num_lrs=${#LEARNING_RATES[@]}

# Loop from 0 to num_lrs-1
for (( i=0; i<${num_lrs}; i++ )); do

    # Get current learning rates from arrays
    CURRENT_LR=${LEARNING_RATES[$i]}
    CURRENT_TEXT_LR=${TEXT_ENCODER_LRS[$i]}

    # Construct config name and temporary training config file path
    # Use LR value directly in name for clarity, replacing dots with 'p' if needed for filename safety
    LR_NAME=$(echo $CURRENT_LR | sed 's/\./p/') # e.g., 0.0002 -> 0p0002
    CONFIG_NAME="vera_lr_${LR_NAME}"
    TEMP_TRAINING_CONFIG_PATH="./temp_training_config_${CONFIG_NAME}.yaml" # Relative to PROJECT_DIR

    echo ""
    echo "***************************************"
    echo "Processing Learning Rate index: ${i}"
    echo "Current LR: ${CURRENT_LR}"
    echo "Current Text LR: ${CURRENT_TEXT_LR}"
    echo "Config Name:     ${CONFIG_NAME}"
    echo "Temp Config File: ${TEMP_TRAINING_CONFIG_PATH}"
    echo "***************************************"

    # --- Generate Temporary Training Config --- #
    echo "Generating temporary training config..."
    # Use sed to replace the learning rate lines in the base config
    sed -e "s/^learning_rate:.*/learning_rate: ${CURRENT_LR}/" \
        -e "s/^text_encoder_lr:.*/text_encoder_lr: ${CURRENT_TEXT_LR}/" \
        "${BASE_TRAINING_CONFIG_PATH}" > "${TEMP_TRAINING_CONFIG_PATH}"

    if [ ! -f "$TEMP_TRAINING_CONFIG_PATH" ]; then
        echo "Error: Failed to create temporary training config file: ${TEMP_TRAINING_CONFIG_PATH}" >&2
        continue # Skip to the next learning rate
    fi
    # Verify the change (optional but helpful for debugging)
    echo "Generated temp config contents (first 20 lines):"
    head -n 20 "${TEMP_TRAINING_CONFIG_PATH}"
    echo "..."

    # Construct the specific output directory for this training run (relative to project dir)
    TRAINING_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
    # Construct the absolute path for checking existence and passing to python
    ABS_TRAINING_OUTPUT_DIR="${PROJECT_DIR}/${TRAINING_OUTPUT_DIR}"

    echo "======================================="
    echo "Configuration for LR ${CURRENT_LR}"
    echo "Adapter Config: ${ADAPTER_CONFIG_PATH}" # Fixed adapter config
    echo "Training Config (Temp): ${TEMP_TRAINING_CONFIG_PATH}"
    echo "Evaluation Config: ${EVALUATION_CONFIG_PATH}"
    echo "Output Dir:     ${ABS_TRAINING_OUTPUT_DIR}"
    echo "======================================="

    # --- Process Object Instances for current LR --- #
    echo "======================================="
    echo "Processing Object Instances (LR ${CURRENT_LR})"
    echo "======================================="

    for INSTANCE in "${OBJECT_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (LR ${CURRENT_LR})"
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
            echo "Skipping training for ${INSTANCE} (LR ${CURRENT_LR})."
        else
            echo "Starting VeRA Training Run for ${INSTANCE} (LR ${CURRENT_LR})..."
            # Ensure the instance output directory exists
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            # Run training with instance data directory
            # Pass *temporary* training_config, fixed adapter_config, and instance specifics
            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TEMP_TRAINING_CONFIG_PATH" \
                --adapter_config "$ADAPTER_CONFIG_PATH" \
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
            echo "Training for ${INSTANCE} (LR ${CURRENT_LR}) finished successfully."
        fi

        # --- Evaluation --- #
        # Check if evaluation results already exist for this instance
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")

        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            # If results found, skip evaluation
            echo "======================================="
            echo "Skipping Evaluation for ${INSTANCE} (LR ${CURRENT_LR})"
            echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
            echo "======================================="
        else
            # If no results found, proceed with evaluation
            echo "======================================="
            echo "Starting Evaluation for ${INSTANCE} (LR ${CURRENT_LR})"
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
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (LR ${CURRENT_LR})." >&2
                # Continue to next instance instead of exiting
                continue
            fi

            # Run evaluation using the trained adapter
            # Pass the default experiment config, specific adapter dir, and instance info
            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EVALUATION_CONFIG_PATH" \
              --pretrained_model_path "$PRETRAINED_MODEL" \
              --output_dir "$EVALUATION_OUTPUT_DIR" \
              --prompts_file "$OBJECT_PROMPTS_FILE" \
              --adapter_dirs "$ADAPTER_DIR" \
              --num_images "$NUM_IMAGES_PER_PROMPT" \
              --max_prompts "$MAX_PROMPTS" \
              ${USE_UNIQUE_SEEDS:+--use_unique_seeds} \
              --concepts "$INSTANCE" \
              --seed "$SEED" \
              --device cuda \
              --mixed_precision "$MIXED_PRECISION" \
              --num_inference_steps "$INFERENCE_STEPS" \
              --guidance_scale "$GUIDANCE_SCALE" \
              --unique_token "$UNIQUE_TOKEN"

            # Check if evaluation actually produced images
            if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
                echo "Error: Evaluation finished but no image files (.png/.jpg) were found in ${EVALUATION_OUTPUT_DIR}" >&2
            else
                echo "Evaluation complete for ${INSTANCE} (LR ${CURRENT_LR})! Results saved to ${EVALUATION_OUTPUT_DIR}"
            fi
        fi
    done

    # --- Process Live Instances for current LR --- #
    echo ""
    echo "======================================="
    echo "Processing Live Instances (LR ${CURRENT_LR})"
    echo "======================================="

    for INSTANCE in "${LIVE_INSTANCES[@]}"; do
        echo ""
        echo "======================================="
        echo "Processing Instance: ${INSTANCE} (LR ${CURRENT_LR})"
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
            echo "Skipping training for ${INSTANCE} (LR ${CURRENT_LR})."
        else
            echo "Starting VeRA Training Run for ${INSTANCE} (LR ${CURRENT_LR})..."
            # Ensure the instance output directory exists
            mkdir -p "$INSTANCE_OUTPUT_DIR"

            # Run training with instance data directory
            python -m sdxl_dreambooth.train_sdxl \
                --training_config "$TEMP_TRAINING_CONFIG_PATH" \
                --adapter_config "$ADAPTER_CONFIG_PATH" \
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
            echo "Training for ${INSTANCE} (LR ${CURRENT_LR}) finished successfully."
        fi

        # --- Evaluation --- #
        # Check if evaluation results already exist for this instance
        EVAL_DIRS=$(find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name "evaluation_results_*")

        if [ -n "$EVAL_DIRS" ] && [ -n "$(find $EVAL_DIRS -type f -name "*.png" -o -name "*.jpg" | head -n 1)" ]; then
            # If results found, skip evaluation
            echo "======================================="
            echo "Skipping Evaluation for ${INSTANCE} (LR ${CURRENT_LR})"
            echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
            echo "======================================="
        else
            # If no results found, proceed with evaluation
            echo "======================================="
            echo "Starting Evaluation for ${INSTANCE} (LR ${CURRENT_LR})"
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
                echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE} (LR ${CURRENT_LR})." >&2
                # Continue to next instance instead of exiting
                continue
            fi

            # Run evaluation using the trained adapter with LIVE prompts file
            python -m sdxl_dreambooth.run_evaluation \
              --config_path "$EVALUATION_CONFIG_PATH" \
              --pretrained_model_path "$PRETRAINED_MODEL" \
              --output_dir "$EVALUATION_OUTPUT_DIR" \
              --prompts_file "$LIVE_PROMPTS_FILE" \
              --adapter_dirs "$ADAPTER_DIR" \
              --num_images "$NUM_IMAGES_PER_PROMPT" \
              --max_prompts "$MAX_PROMPTS" \
              ${USE_UNIQUE_SEEDS:+--use_unique_seeds} \
              --concepts "$INSTANCE" \
              --seed "$SEED" \
              --device cuda \
              --mixed_precision "$MIXED_PRECISION" \
              --num_inference_steps "$INFERENCE_STEPS" \
              --guidance_scale "$GUIDANCE_SCALE" \
              --unique_token "$UNIQUE_TOKEN"

            # Check if evaluation actually produced images
            if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
                echo "Error: Evaluation finished but no image files (.png/.jpg) were found in ${EVALUATION_OUTPUT_DIR}" >&2
            else
                echo "Evaluation complete for ${INSTANCE} (LR ${CURRENT_LR})! Results saved to ${EVALUATION_OUTPUT_DIR}"
            fi
        fi
    done

    # --- Clean up temporary training config file --- #
    echo "Removing temporary training config file: ${TEMP_TRAINING_CONFIG_PATH}"
    rm -f "${TEMP_TRAINING_CONFIG_PATH}"

done # End of loop for learning rates

echo ""
echo "======================================="
echo "All VeRA learning rate multipliers processed successfully"
echo "Outputs saved in ${PROJECT_DIR}/${BASE_OUTPUT_DIR}"
echo "=======================================" 