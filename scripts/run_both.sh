CUDA_VISIBLE_DEVICES=1 python run.py \
    --out_dir "./outputs/" \
    --mode "both" \
    --source_path "./example_images/church/app/01.jpg" "./example_images/church/struct/01.jpg" \
    --source_domain "church" "church" \
    --target_prompt "a photo of a church" \
    --struct_end_step 30 \
    --threshold 0.05 \
    --latent_blend_type "non"