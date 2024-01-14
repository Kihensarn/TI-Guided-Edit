import argparse
import os
import json
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.io import read_image
from diffusers import DDIMScheduler
from utils.diffuser_utils import TIGuidedPipeline
from utils.masactrl_utils import regiter_attention_editor_diffusers, AttentionStore, register_conv_control_efficient
from utils.masactrl import UnifiedSelfAttentionControl
from pytorch_lightning import seed_everything


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = T.Resize(512)(image)
    image = T.CenterCrop(512)(image)
    image = image.to(device)
    return image

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return out

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters')

    # path config
    parser.add_argument('--model_path', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--out_dir', type=str, default="./outputs/")

    # image config
    parser.add_argument('--source_path', nargs = '+', required=True, type=str, default=None)
    parser.add_argument('--source_domain', nargs = '+', required=True, type=str, default=None)
    parser.add_argument('--source_prompt', nargs = '+', type=str, default=None, help="source prompt for inversion, automatically formatted from the source domain if not provided.")
    parser.add_argument('--target_prompt', required=True, type=str, default="")
    parser.add_argument('--negative_prompt', type=str, default="ugly, blurry, black, low res, unrealistic")
    
    # inference config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5)
    parser.add_argument('--inv_scale', type=float, default=1)
    parser.add_argument('--text_scale', type=float, default=7.5)
    parser.add_argument('--latent_blend_type', type=str, default="non", help="bg and fg means blend background and foreground latents from source image, respectively.")
    parser.add_argument('--latent_blend_step', type=int, default=40)
    parser.add_argument('--adain_start_step', type=int, default=10)
    parser.add_argument('--adain_end_step', type=int, default=30)
    parser.add_argument('--mode', type=str, default="app", help="editing mode: app means no-rigid editing, struct means rigid editing, and both means image-based editing.")
    
    # mask collection config
    parser.add_argument('--res', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.05, help="if the automatically generated mask is not accurate, you can adjust it between 0.02 and 0.15.")
    parser.add_argument('--save_mask_timestep', type=int, default=10)
    
    # unified self-attn config
    parser.add_argument('--conv_injection_t', type=int, default=40, help="feature injection steps")
    parser.add_argument('--app_start_step', type=int, default=4)
    parser.add_argument('--app_end_step', type=int, default=40)
    parser.add_argument('--app_start_layer', type=int, default=10)
    parser.add_argument('--struct_start_step', type=int, default=0)
    parser.add_argument('--struct_end_step', type=int, default=25)
    parser.add_argument('--struct_start_layer', type=int, default=8)
    parser.add_argument('--contrast_strength', type=float, default=1.67)
    parser.add_argument('--injection_step', type=int, default=1, help="interval steps for simultaneously injecting app and structural information.")

    args = parser.parse_args()
    return args

def sample(args):
    seed_everything(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # init model
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = TIGuidedPipeline.from_pretrained(args.model_path, scheduler=scheduler, cross_attention_kwargs={"scale": 0.1}, torch_dtype=torch.float32).to(device)
    
    # set output directory
    sub_dir = f"{args.mode}_{'_'.join(args.source_domain)}_{args.target_prompt.replace(' ', '_')}"
    output_dir = os.path.join(args.out_dir, sub_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # set source prompt for inversion
    if args.source_prompt is None:
        args.source_prompt = [f"a {domain_name}" for domain_name in args.source_domain]

    # set initial flag
    appearance_invert_flag = False
    struct_invert_flag = False
    
    # set parameters for each editng mode
    if args.mode == "app":
        appearance_image_path = args.source_path[0]
        appearance_domain_name = args.source_domain[0]
        appearance_prompt = args.source_prompt[0]
        struct_prompt = args.target_prompt
        appearance_invert_flag = True
    elif args.mode == "struct":
        struct_image_path = args.source_path[0]
        struct_domain_name = args.source_domain[0]
        struct_prompt = args.source_prompt[0]
        appearance_prompt = args.target_prompt
        struct_invert_flag = True
    elif args.mode == "both":
        appearance_image_path = args.source_path[0]
        appearance_domain_name = args.source_domain[0]
        appearance_prompt = args.source_prompt[0]
        appearance_invert_flag = True

        struct_image_path = args.source_path[1]
        struct_domain_name = args.source_domain[1]
        struct_prompt = args.source_prompt[1]
        struct_invert_flag = True
    else:
        raise NotImplementedError

    # save parameters
    args_dict = vars(args)
    output_file = 'arguments.json'
    with open(os.path.join(output_dir, output_file), 'w') as json_file:
        json.dump(args_dict, json_file, indent=2)
    print(f'Arguments saved to {output_file}')

    # set prompt
    prompts = [appearance_prompt, args.target_prompt, struct_prompt]
    appearance_neg_prompt = appearance_prompt if appearance_invert_flag else args.negative_prompt
    struct_neg_prompt = struct_prompt if struct_invert_flag else args.negative_prompt
    neg_prompts = [appearance_neg_prompt, args.negative_prompt, struct_neg_prompt]

    # set scale
    app_scale = args.inv_scale if appearance_invert_flag else args.text_scale
    struct_scale = args.inv_scale if struct_invert_flag else args.text_scale
    scale = [app_scale, args.guidance_scale, struct_scale]

    # invert the reference images
    if struct_invert_flag:
        ind = get_word_inds(struct_prompt, struct_domain_name, model.tokenizer)
        assert len(ind) != 0, "The object name must in the source prompt."

        struct_image = load_image(struct_image_path, device)
        editor = AttentionStore(res=args.res, ref_token_idx=ind, save_mask_timestep=args.save_mask_timestep, 
                                threshold=args.threshold, save_dir=output_dir, image_name="struct")
        regiter_attention_editor_diffusers(model, editor)

        inv_start_code_struct, latents_list_struct = model.invert(struct_image,
                                                    struct_prompt,
                                                    guidance_scale=args.inv_scale,
                                                    num_inference_steps=args.num_inference_steps,
                                                    return_intermediates=True)
        mask_struct = editor.get_aggregate_mask()
        model.set_struct_mask(mask_struct)
        
    if appearance_invert_flag:
        ind = get_word_inds(appearance_prompt, appearance_domain_name, model.tokenizer)
        assert len(ind) != 0, "The domain name must in the source prompt."

        appearance_image = load_image(appearance_image_path, device)
        editor = AttentionStore(res=args.res, ref_token_idx=ind, save_mask_timestep=args.save_mask_timestep, 
                                threshold=args.threshold, save_dir=output_dir, image_name="app")
        regiter_attention_editor_diffusers(model, editor)

        inv_start_code_appearance, latents_list_appearance = model.invert(appearance_image,
                                                    appearance_prompt,
                                                    guidance_scale=args.inv_scale,
                                                    num_inference_steps=args.num_inference_steps,
                                                    return_intermediates=True)
        mask_appearance = editor.get_aggregate_mask()
        model.set_app_mask(mask_appearance)

    # set the start code
    if args.mode == "app":
        start_code = torch.cat([inv_start_code_appearance, 
                                inv_start_code_appearance, 
                                inv_start_code_appearance], dim=0)
    elif args.mode == "struct":
        start_code = torch.cat([inv_start_code_struct, 
                                inv_start_code_struct, 
                                inv_start_code_struct], dim=0)
    elif args.mode == "both":
        start_code = torch.cat([inv_start_code_appearance, 
                                inv_start_code_struct, 
                                inv_start_code_struct], dim=0)
    else:
        raise NotImplementedError

    # hijack the attention module using multi mixup
    editor = UnifiedSelfAttentionControl(appearance_start_step=args.app_start_step, 
                                        appearance_end_step=args.app_end_step, 
                                        appearance_start_layer=args.app_start_layer, 
                                        struct_start_step=args.struct_start_step, 
                                        struct_end_step=args.struct_end_step, 
                                        struct_start_layer=args.struct_start_layer, 
                                        mix_type=args.mode, 
                                        contrast_strength=args.contrast_strength, 
                                        injection_step=args.injection_step)
    regiter_attention_editor_diffusers(model, editor)

    # hijack the resblock module 
    injection_step = args.injection_step if args.mode == "app" else 1
    conv_injection_timesteps = scheduler.timesteps[:args.conv_injection_t:injection_step] if args.conv_injection_t >= 0 else []
    register_conv_control_efficient(model, conv_injection_timesteps)

    # inference the synthesized image
    image_results = model(prompts,
                        latents=start_code,
                        num_inference_steps=args.num_inference_steps,
                        scale=scale,
                        neg_prompts=neg_prompts,
                        ref_intermediate_latents_app=latents_list_appearance if appearance_invert_flag else None,
                        ref_intermediate_latents_struct=latents_list_struct if struct_invert_flag else None,
                        callback=model.get_adain_callback(args.adain_start_step, args.adain_end_step) if args.mode == "both" else None,
                        latent_blend_type=args.latent_blend_type,
                        latent_blend_step=args.latent_blend_step,
    )

    # save results
    if args.mode == "both":
        joint_image = torch.cat([appearance_image * 0.5 + 0.5, struct_image * 0.5 + 0.5, image_results[1].unsqueeze(0)], dim=0)
    else:
        source_image = appearance_image if appearance_invert_flag else struct_image
        joint_image = torch.cat([source_image * 0.5 + 0.5, image_results[1].unsqueeze(0)], dim=0)
    save_image(image_results[1].unsqueeze(0), os.path.join(output_dir, f"result.jpg"))
    save_image(joint_image, os.path.join(output_dir, f"joint_result.jpg"))
    print("Syntheiszed images are saved in", output_dir)


if __name__ == "__main__":
    args = parse_args()
    sample(args)