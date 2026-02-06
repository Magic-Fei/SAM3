# SAM 3: Segment Anything with Concepts

Meta Superintelligence Labs

For full author list and details, see the [paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).

[[`Paper`](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)]
[[`Project`](https://ai.meta.com/sam3)]
[[`Demo`](https://segment-anything.com/)]
[[`Blog`](https://ai.meta.com/blog/segment-anything-model-3/)]
<!-- [[`BibTeX`](#citing-sam-3)] -->

SAM 3 is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. Compared to its predecessor [SAM 2](https://github.com/facebookresearch/sam2), SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars. Unlike prior work, SAM 3 can handle a vastly larger set of open-vocabulary prompts. It achieves 75-80% of human performance on our new [SA-CO benchmark](https://github.com/facebookresearch/sam3?tab=readme-ov-file#sa-co-dataset) which contains 270K unique concepts, over 50 times more than existing benchmarks.

This breakthrough is driven by an innovative data engine that has automatically annotated over 4 million unique concepts, creating the largest high-quality open-vocabulary segmentation dataset to date. In addition, SAM 3 introduces a new model architecture featuring a presence token that improves discrimination between closely related text prompts (e.g., "a player in white" vs. "a player in red"), as well as a decoupled detector–tracker design that minimizes task interference and scales efficiently with data.

## Installation

### Prerequisites

- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher

1. **Create a new Conda environment:**

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

2. **Install PyTorch with CUDA support:**

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. **Clone the repository and install the package:**

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

4. **Install additional dependencies for example notebooks or development:**

```bash
# For running example notebooks
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"
```

## Getting Started

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### Basic Usage

```python
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0, # Arbitrary frame index
        text="<YOUR_TEXT_PROMPT>",
    )
)
output = response["outputs"]
```

## Examples

The `examples` directory contains notebooks demonstrating how to use SAM3 with various types of prompts. To run the examples:

```bash
pip install -e ".[notebooks]"
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## Training on Custom Dataset

This section provides a guide for training SAM3 on your own single-class segmentation dataset.

### Step 1: Convert Dataset Format

#### 1.1 Install Dependencies

```bash
pip install labelme pycocotools pillow numpy tqdm
```

#### 1.2 Prepare Labelme Dataset

Ensure your dataset structure is as follows:
```
labelme_dataset/
├── image1.jpg
├── image1.json
├── image2.jpg
├── image2.json
└── ...
```

#### 1.3 Run Conversion Script

```bash
python labelme_to_coco.py \
    --labelme_dir /path/to/labelme_dataset \
    --output_dir /path/to/coco_dataset \
    --class_name "your_class_name" \
    --train_split 0.8
```

**Parameters:**
- `--labelme_dir`: Labelme annotation directory (containing JSON and images)
- `--output_dir`: Output COCO format dataset directory
- `--class_name`: Class name (e.g., "person", "car", "dog")
- `--train_split`: Training set ratio (default 0.8, i.e., 80% train, 20% validation)

**Example:**
```bash
python labelme_to_coco.py \
    --labelme_dir ./my_labelme_data \
    --output_dir ./coco_dataset \
    --class_name "object" \
    --train_split 0.8
```

#### 1.4 Converted Dataset Structure

```
coco_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations.json
└── val/
    ├── images/
    │   ├── image3.jpg
    │   ├── image4.jpg
    │   └── ...
    └── annotations.json
```

### Step 2: Configure Training Parameters

#### 2.1 Edit Configuration File

Edit `sam3/train/conf/train_config_single_class.yaml` and modify the following paths:

```yaml
paths:
  dataset_root: /path/to/coco_dataset  # Change to your dataset path
  experiment_log_dir: /path/to/experiments  # Change to experiment output directory
  bpe_path: assets/bpe_simple_vocab_16e6.txt.gz  # BPE vocabulary path

dataset:
  class_name: "your_class_name"  # Change to your class name
```

#### 2.2 Adjust Training Parameters (Optional)

Based on your GPU memory and dataset size, you can adjust:

```yaml
scratch:
  train_batch_size: 2  # Can be reduced to 1 if GPU memory is insufficient
  val_batch_size: 1
  num_train_workers: 4  # Number of data loading threads
  resolution: 1008  # Image resolution, can be reduced to save memory
  max_epochs: 50  # Number of training epochs
```

### Step 3: Start Training

#### 3.1 Single GPU Training

```bash
python sam3/train/train.py -c sam3/train/conf/train_config_single_class.yaml
```

#### 3.2 Multi-GPU Training

If you have multiple GPUs, modify `gpus_per_node` in the configuration file:

```yaml
launcher:
  gpus_per_node: 2  # Change to your number of GPUs
```

Then run:
```bash
python sam3/train/train.py -c sam3/train/conf/train_config_single_class.yaml
```

#### 3.3 Resume Training from Checkpoint

If training is interrupted, you can resume from a checkpoint:

```yaml
trainer:
  checkpoint:
    resume_from: /path/to/checkpoint.pt  # Specify checkpoint path
```

### Step 4: Monitor Training

#### 4.1 TensorBoard

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir /path/to/experiments/tensorboard
```

#### 4.2 View Logs

Training logs are saved in:
```
/path/to/experiments/logs/
```

### Step 5: Verify Training Results

After training, checkpoints are saved in:
```
/path/to/experiments/checkpoints/
```

Validation results are saved in:
```
/path/to/experiments/dumps/val/
```

### Training Tips

1. **Model Compression**: After training, you can compress the model to save space:
   ```bash
   python tools/compress_model.py
   ```
   This will extract weights and convert to FP16, reducing model size by ~60%.

2. **Save FP16 Models Directly**: You can configure training to save FP16 models directly:
   ```yaml
   checkpoint:
     save_model_only: true  # Only save model weights
     save_fp16: true  # Save in FP16 format
     save_epochs: [10, 20, 30, 40, 50]  # Specify epochs to save
   ```

3. **Batch Inference**: After training, use the trained model for batch inference:
   ```bash
   python tools/batch_inference.py
   ```
   See [BATCH_INFERENCE_README.md](tools/BATCH_INFERENCE_README.md) for details.

### Common Issues

#### Q1: Out of Memory (OOM)

**Solutions:**
1. Reduce `train_batch_size` (change to 1)
2. Lower `resolution` (change to 800 or smaller)
3. Reduce `num_train_workers`

#### Q2: Slow Training Speed

**Solutions:**
1. Increase `num_train_workers`
2. Use multi-GPU training
3. Reduce `resolution`

#### Q3: Conversion Script Errors

**Possible causes:**
1. Incorrect Labelme JSON format
2. Image paths not found
3. Polygon annotation format issues

**Solutions:**
- Check Labelme JSON file format
- Ensure image files exist
- Ensure annotations are in polygon format

#### Q4: Class Name Mismatch

**Solution:**
Ensure the `class_name` in the configuration file matches the name used in the conversion script.

### Complete Example Commands

```bash
# 1. Convert dataset
python labelme_to_coco.py \
    --labelme_dir ./my_labelme_data \
    --output_dir ./coco_dataset \
    --class_name "object" \
    --train_split 0.8

# 2. Edit paths in configuration file

# 3. Start training
python sam3/train/train.py -c sam3/train/conf/train_config_single_class.yaml

# 4. Monitor training
tensorboard --logdir ./experiments/tensorboard

# 5. Compress model (after training)
python tools/compress_model.py

# 6. Batch inference
python tools/batch_inference.py
```

### Notes

1. **Class Name**: The class name used during training and inference must be consistent
2. **Image Format**: Supports common image formats (jpg, png, etc.)
3. **Annotation Format**: Only supports polygon annotations, not rectangles or other shapes
4. **Pretrained Model**: First training will automatically download the pretrained model from HuggingFace, requires internet connection
5. **GPU Requirements**: Recommended to use GPU with at least 16GB VRAM

For more detailed training documentation, see [训练指南.md](训练指南.md).

## Model

SAM 3 consists of a detector and a tracker that share a vision encoder. It has 848M parameters. The
detector is a DETR-based model conditioned on text, geometry, and image
exemplars. The tracker inherits the SAM 2 transformer encoder-decoder
architecture, supporting video segmentation and interactive refinement.

## Results

SAM 3 achieves state-of-the-art performance on various benchmarks:

**Image Results:**
- SA-Co/Gold: 54.1 cgF1 (Instance Segmentation), 55.7 cgF1 (Box Detection)
- LVIS: 37.2 cgF1 (Instance Segmentation), 40.6 cgF1 (Box Detection)
- COCO: 56.4 AP (Box Detection)

**Video Results:**
- SA-V test: 30.3 cgF1, 58.0 pHOTA
- YT-Temporal-1B test: 50.8 cgF1, 69.9 pHOTA
- SmartGlasses test: 36.4 cgF1, 63.6 pHOTA

## SA-Co Dataset

We release 2 image benchmarks, [SA-Co/Gold](scripts/eval/gold/README.md) and
[SA-Co/Silver](scripts/eval/silver/README.md), and a video benchmark
[SA-Co/VEval](scripts/eval/veval/README.md). The datasets contain images (or videos) with annotated noun phrases. Each image/video and noun phrase pair is annotated with instance masks and unique IDs of each object matching the phrase.

* HuggingFace host: [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold), [SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver) and [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)
* Roboflow host: [SA-Co/Gold](https://universe.roboflow.com/sa-co-gold), [SA-Co/Silver](https://universe.roboflow.com/sa-co-silver) and [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

## Development

To set up the development environment:

```bash
pip install -e ".[dev,train]"
```

To format the code:

```bash
ufmt format .
```

## Contributing

See [contributing](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file
for details.

## Acknowledgements

We would like to thank the following people for their contributions to the SAM 3 project: Alex He, Alexander Kirillov,
Alyssa Newcomb, Ana Paula Kirschner Mofarrej, Andrea Madotto, Andrew Westbury, Ashley Gabriel, Azita Shokpour,
Ben Samples, Bernie Huang, Carleigh Wood, Ching-Feng Yeh, Christian Puhrsch, Claudette Ward, Daniel Bolya,
Daniel Li, Facundo Figueroa, Fazila Vhora, George Orlin, Hanzi Mao, Helen Klein, Hu Xu, Ida Cheng, Jake Kinney,
Jiale Zhi, Jo Sampaio, Joel Schlosser, Justin Johnson, Kai Brown, Karen Bergan, Karla Martucci, Kenny Lehmann,
Maddie Mintz, Mallika Malhotra, Matt Ward, Michelle Chan, Michelle Restrepo, Miranda Hartley, Muhammad Maaz,
Nisha Deo, Peter Park, Phillip Thomas, Raghu Nayani, Rene Martinez Doehner, Robbie Adkins, Ross Girshik, Sasha
Mitts, Shashank Jain, Spencer Whitehead, Ty Toledano, Valentin Gabeur, Vincent Cho, Vivian Lee, William Ngan,
Xuehai He, Yael Yungster, Ziqi Pang, Ziyi Dou, Zoe Quake.

<!-- ## Citing SAM 3

If you use SAM 3 or the SA-Co dataset in your research, please use the following BibTeX entry.

```bibtex
TODO
``` -->
