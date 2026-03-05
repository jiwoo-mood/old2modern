#bash inference.sh 로 실행

#캐시 경로는 서버 번호에 맞게 변경해주세요
#r6, r7의 경우 /mnt/lustre/users/dongyun/ckpt/FLUX.1-Kontext-dev-12B
#r8의 경우 /mnt/ssd1/users/jiwoo/FLUX.1-Kontext-dev-12B
CACHE_DIR="/workspace/FLUX.1-Kontext-dev-12B"

PROMPTS=(
    " ", #single prompt로 해도 되고,
    " " #여러 prompt list로 넣어도 됨. 
)

#negative prompt 고정
NEGATIVE_PROMPT="upscaling, resize, crop, added objects, changed composition, geometry changes, HDR halos, oversharpening, heavy denoise, blur, extra detail" 

INPUT_DIR="data_folder_root"
OUTPUT_DIR="method1_yourname"
GPU=2

mkdir -p "${OUTPUT_DIR}"

#prompt, image, gpu만 건들고 내부 인자&옵션은 건들지 말아주세요

CUDA_VISIBLE_DEVICES=${GPU} accelerate launch --num_processes 1 \
    inference.py \
    --cache_dir "${CACHE_DIR}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt "${PROMPTS[@]}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --local_files_only