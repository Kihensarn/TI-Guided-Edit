CUDA_VISIBLE_DEVICES=1 python run.py \
    --out_dir "./outputs/" \
    --mode "struct" \
    --source_path "./example_images/cat.jpg" \
    --source_domain "cat" \
    --target_prompt "a brown cat" \
    --struct_end_step 25 \
    --threshold 0.05 \
    --latent_blend_type "bg" 