# --- Configuration --- #
# Define the specific adapter config for this run
ADAPTER_CONFIG_PATH="./sdxl_dreambooth/config/waveft_config.yaml"
# Base directory for all outputs
BASE_OUTPUT_DIR="./outputs"

# --- Derived Paths --- #
PROJECT_DIR=$(pwd) # Capture project dir early

CONFIG_NAME=$(basename "$(dirname "$ADAPTER_CONFIG_PATH")")

# Construct the specific output directory for this training run (relative to project dir)
TRAINING_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}"
# Construct the absolute path for checking existence and passing to python
ABS_TRAINING_OUTPUT_DIR="${PROJECT_DIR}/${TRAINING_OUTPUT_DIR}"

# Use a single base training config, specific LR/steps will come from adapter_config.yaml
TRAINING_CONFIG_PATH="./sdxl_dreambooth/config/training_config.yaml"

# Set Hugging Face cache directory to be local to the project
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

echo "======================================="
echo "Main Configuration"
echo "Adapter Config: ${ADAPTER_CONFIG_PATH}"
echo "Base Training Config: ${TRAINING_CONFIG_PATH}" # Note: LR/Steps from adapter config override this
echo "Config Name:    ${CONFIG_NAME}"
echo "Output Dir:     ${ABS_TRAINING_OUTPUT_DIR}"
echo "======================================="

# To test all instances uncomment the following lines
# OBJECT_INSTANCES=(
#     "backpack" "backpack_dog" "bear_plushie" "berry_bowl" "can" "candle" 
#     "clock" "colorful_sneaker" "duck_toy" "fancy_boot" "grey_sloth_plushie" 
#     "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" 
#     "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie"
# )
OBJECT_INSTANCES=()
OBJECT_PROMPTS_FILE="./dataset/prompts_and_classes_obj.txt"

# To test all live instances uncomment the following lines
#LIVE_INSTANCES=(
#     "cat" "cat2" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8"
# )
LIVE_INSTANCES=("dog")
LIVE_PROMPTS_FILE="./dataset/prompts_and_classes_live.txt" 

# Set the unique token for all training and evaluation
UNIQUE_TOKEN="zjw"

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
NUM_IMAGES_PER_PROMPT=4
MAX_PROMPTS=25
SEED=42
GUIDANCE_SCALE=7.5
INFERENCE_STEPS=30
MIXED_PRECISION="bf16"
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
CONFIG_PATH="./sdxl_dreambooth/config/experiment_config.yaml"

# Process object instances
echo "======================================="
echo "Processing Object Instances"
echo "======================================="

for INSTANCE in "${OBJECT_INSTANCES[@]}"; do
    echo ""
    echo "======================================="
    echo "Processing Instance: ${INSTANCE}"
    echo "======================================="
    
    # Get the correct class name for this instance
    CLASS_NAME=$(get_class_name "$INSTANCE" "$OBJECT_PROMPTS_FILE")
    echo "Instance: ${INSTANCE}, Class: ${CLASS_NAME}"
    
    # Set instance-specific paths
    INSTANCE_OUTPUT_DIR="${ABS_TRAINING_OUTPUT_DIR}/${INSTANCE}"
    INSTANCE_DATA_DIR="./dataset/${INSTANCE}"
    INSTANCE_MODEL_DIR="${INSTANCE_OUTPUT_DIR}/final_model"
    
    # Check if the final model directory already exists for this instance
    if [ -d "${INSTANCE_MODEL_DIR}" ]; then
        echo "Final model directory already exists: ${INSTANCE_MODEL_DIR}"
        echo "Skipping training for ${INSTANCE}."
    else
        echo "Starting Training Run for ${INSTANCE}..."
        # Ensure the instance output directory exists
        mkdir -p "$INSTANCE_OUTPUT_DIR"

        # Run training with instance data directory
        python -m sdxl_dreambooth.train_sdxl \
            --training_config "$TRAINING_CONFIG_PATH" \
            --adapter_config "$ADAPTER_CONFIG_PATH" \
            --experiment_config ./sdxl_dreambooth/config/experiment_config.yaml \
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
        echo "Training for ${INSTANCE} finished successfully."
    fi

    # --- Evaluation --- #
    # Check if evaluation results already exist for this instance (any subdir containing "eval")
    if find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name '*eval*' -print -quit | grep -q .; then
        # If results found, skip evaluation
        echo "======================================="
        echo "Skipping Evaluation for ${INSTANCE}"
        echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
        echo "======================================="
    else
        # If no results found, proceed with evaluation
        echo "======================================="
        echo "Starting Evaluation for ${INSTANCE}"
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
            echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE}." >&2
            # Continue to next instance instead of exiting
            continue
        fi

        # Run evaluation using the trained adapter
        python -m sdxl_dreambooth.run_evaluation \
          --config_path "$CONFIG_PATH" \
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

        # Check if evaluation actually produced images
        if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 2 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
            echo "Error: Evaluation finished but no image files (.png/.jpg) were found in ${EVALUATION_OUTPUT_DIR}" >&2
        else
            echo "Evaluation complete for ${INSTANCE}! Results saved to ${EVALUATION_OUTPUT_DIR}"
        fi
    fi
done

# Process live instances
echo ""
echo "======================================="
echo "Processing Live Instances"
echo "======================================="

for INSTANCE in "${LIVE_INSTANCES[@]}"; do
    echo ""
    echo "======================================="
    echo "Processing Instance: ${INSTANCE}"
    echo "======================================="
    
    # Get the correct class name for this instance
    CLASS_NAME=$(get_class_name "$INSTANCE" "$LIVE_PROMPTS_FILE")
    echo "Instance: ${INSTANCE}, Class: ${CLASS_NAME}"
    
    # Set instance-specific paths
    INSTANCE_OUTPUT_DIR="${ABS_TRAINING_OUTPUT_DIR}/${INSTANCE}"
    INSTANCE_DATA_DIR="./dataset/${INSTANCE}"
    INSTANCE_MODEL_DIR="${INSTANCE_OUTPUT_DIR}/final_model"
    
    # Check if the final model directory already exists for this instance
    if [ -d "${INSTANCE_MODEL_DIR}" ]; then
        echo "Final model directory already exists: ${INSTANCE_MODEL_DIR}"
        echo "Skipping training for ${INSTANCE}."
    else
        echo "Starting Training Run for ${INSTANCE}..."
        # Ensure the instance output directory exists
        mkdir -p "$INSTANCE_OUTPUT_DIR"

        # Run training with instance data directory
        python -m sdxl_dreambooth.train_sdxl \
            --training_config "$TRAINING_CONFIG_PATH" \
            --adapter_config "$ADAPTER_CONFIG_PATH" \
            --experiment_config ./sdxl_dreambooth/config/experiment_config.yaml \
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
        echo "Training for ${INSTANCE} finished successfully."
    fi

    # --- Evaluation --- #
    # Check if evaluation results already exist for this instance (any subdir containing "eval")
    if find "${INSTANCE_OUTPUT_DIR}" -maxdepth 1 -type d -name '*eval*' -print -quit | grep -q .; then
        # If results found, skip evaluation
        echo "======================================="
        echo "Skipping Evaluation for ${INSTANCE}"
        echo "Found existing evaluation results in: ${INSTANCE_OUTPUT_DIR}"
        echo "======================================="
    else
        # If no results found, proceed with evaluation
        echo "======================================="
        echo "Starting Evaluation for ${INSTANCE}"
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
            echo "Error: Adapter directory ${ADAPTER_DIR} not found. Cannot run evaluation for ${INSTANCE}." >&2
            # Continue to next instance instead of exiting
            continue
        fi

        # Run evaluation using the trained adapter with LIVE prompts file
        python -m sdxl_dreambooth.run_evaluation \
          --config_path "$CONFIG_PATH" \
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

        # Check if evaluation actually produced images
        if ! find "$EVALUATION_OUTPUT_DIR" -maxdepth 2 -type f \( -name '*.png' -o -name '*.jpg' \) -print -quit | grep -q .; then
            echo "Error: Evaluation finished but no image files (.png/.jpg) were found in ${EVALUATION_OUTPUT_DIR}" >&2
        else
            echo "Evaluation complete for ${INSTANCE}! Results saved to ${EVALUATION_OUTPUT_DIR}"
        fi
    fi
done

echo ""
echo "======================================="
echo "All instances processed successfully"
echo "======================================="