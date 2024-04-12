import torch
from diffusers import DiffusionPipeline

model_id = "./model/stable-diffusion-v1-4"
output_path = "./"
print(f"Loading original model... from{model_id}")

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "a photo of kudoshinichi in space"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save(f"{output_path}output_original.png")


print(f"Loading finetuned model... from{model_id}")
state_dict = torch.load("./weight_output/diffusion_pytorch_model.bin")
pipe.unet.load_state_dict(state_dict)

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save(f"{output_path}output.png")
