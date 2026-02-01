"""
Project Configuration - Centralized paths for all scripts
==========================================================
Use this file to maintain consistent paths across the project.
"""

from pathlib import Path

# Root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
REAL_SCREENSHOTS_DIR = DATA_DIR / "real_screenshots"
IMPROVED_DATA_DIR = DATA_DIR / "improved_data"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODEL_PATH = MODELS_DIR / "gui_widget_detection" / "yolov8_training4" / "weights" / "best.pt"
BASE_YOLO_MODEL = MODELS_DIR / "yolov8n.pt"

# Datasets directory
DATASETS_DIR = PROJECT_ROOT / "datasets"
YOLO_DATASET_DIR = DATASETS_DIR / "yolo_dataset"
DATASET_YAML = PROJECT_ROOT / "dataset.yaml"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
GUI_RECREATIONS_DIR = RESULTS_DIR / "gui_recreations"
TEST_OUTPUTS_DIR = RESULTS_DIR / "test_outputs"
REAL_WORLD_RESULTS_DIR = RESULTS_DIR / "real_world"
PRESENTATION_RESULTS_DIR = RESULTS_DIR / "presentation"
RUNS_DIR = RESULTS_DIR / "runs"

# Scripts directories
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAINING_SCRIPTS_DIR = SCRIPTS_DIR / "training"
INFERENCE_SCRIPTS_DIR = SCRIPTS_DIR / "inference"
RECREATION_SCRIPTS_DIR = SCRIPTS_DIR / "recreation"
ANALYSIS_SCRIPTS_DIR = SCRIPTS_DIR / "analysis"
UTILS_SCRIPTS_DIR = SCRIPTS_DIR / "utils"

# Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Documentation directory
DOCS_DIR = PROJECT_ROOT / "docs"

# Default parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_MODEL_SIZE = "yolov8n"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16

# Ensure directories exist
def ensure_directories():
    """Create all necessary directories if they don't exist"""
    dirs = [
        DATA_DIR, ANNOTATIONS_DIR, SCREENSHOTS_DIR, REAL_SCREENSHOTS_DIR,
        MODELS_DIR, DATASETS_DIR, YOLO_DATASET_DIR,
        RESULTS_DIR, GUI_RECREATIONS_DIR, TEST_OUTPUTS_DIR, 
        REAL_WORLD_RESULTS_DIR, PRESENTATION_RESULTS_DIR, RUNS_DIR,
        SCRIPTS_DIR, TRAINING_SCRIPTS_DIR, INFERENCE_SCRIPTS_DIR,
        RECREATION_SCRIPTS_DIR, ANALYSIS_SCRIPTS_DIR, UTILS_SCRIPTS_DIR,
        NOTEBOOKS_DIR, DOCS_DIR
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Print all paths
    print("Project Configuration:")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nData:")
    print(f"  Annotations:      {ANNOTATIONS_DIR}")
    print(f"  Screenshots:      {SCREENSHOTS_DIR}")
    print(f"  Real Screenshots: {REAL_SCREENSHOTS_DIR}")
    print(f"\nModels:")
    print(f"  Trained Model:    {TRAINED_MODEL_PATH}")
    print(f"  Base YOLO:        {BASE_YOLO_MODEL}")
    print(f"\nResults:")
    print(f"  GUI Recreations:  {GUI_RECREATIONS_DIR}")
    print(f"  Test Outputs:     {TEST_OUTPUTS_DIR}")
    print(f"  Real World:       {REAL_WORLD_RESULTS_DIR}")
    print(f"\nAll directories exist: {all(p.exists() for p in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SCRIPTS_DIR])}")
