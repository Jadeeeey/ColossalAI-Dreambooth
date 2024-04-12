import torch
from diffusers import DiffusionPipeline

# model_id = "<Your Model Path>"
model_id = "/home/users/nus/e1216290/scratch/colossalai/ColossalAI/examples/images/dreambooth/weight_output"
print(f"Loading model... from{model_id}")

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of a dog in Starbucks"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("/home/users/nus/e1216290/scratch/colossalai/ColossalAI/examples/images/dreambooth/output.png")
