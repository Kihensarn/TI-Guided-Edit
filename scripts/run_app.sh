CUDA_VISIBLE_DEVICES=1 python run.py \
    --out_dir "./outputs/" \
    --mode "app" \
    --source_path "./example_images/dog.jpg" \
    --source_domain "dog" \
    --target_prompt "a sitting dog" \
    --struct_end_step 25 \
    --threshold 0.05 \
    --latent_blend_type "non"