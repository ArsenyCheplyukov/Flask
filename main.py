# import torch
# from diffusers import StableDiffusionImageVariationPipeline
# from PIL import Image
# from torchvision import transforms
# from torchvision.transforms.functional import to_pil_image
# from transformers import CLIPImageProcessor

# torch.backends.cuda.matmul.allow_tf32 = True

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the diffusion model
# sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
#     "lambdalabs/sd-image-variations-diffusers",
#     revision="2ddbd90b14bc5892c19925b15185e561bc8e5d0a",
#     torch_dtype=torch.float16,
# ).to(device)

# sd_pipe.enable_attention_slicing()
# sd_pipe.enable_sequential_cpu_offload()

# # Load and transform the input image
# im = Image.open("input_image.png")
# original_size = im.size
# new_size = (224, 224)  # Decrease image size
# tform = transforms.Compose(
#     [
#         transforms.Resize(
#             new_size,
#             interpolation=transforms.InterpolationMode.BICUBIC,
#             antialias=False,
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )
# inp = tform(im).unsqueeze(0).to(device)

# # Create the CLIP image processor
# image_processor = CLIPImageProcessor()

# # Generate the output image variations
# variation_images = []
# batch_size = 1  # Decrease batch size
# num_variations = 3
# with torch.no_grad():
#     for i in range(0, num_variations, batch_size):
#         batch_input = inp.repeat(batch_size, 1, 1, 1)
#         # Preprocess the batch_input with CLIPImageProcessor
#         inputs = image_processor(batch_input)
#         out = sd_pipe(inputs.pixel_values, guidance_scale=3)
#         variation_images.extend([v.detach().cpu() for v in out["images"]])

# # Save the variation images
# for idx, variation_image in enumerate(variation_images):
#     variation_image = transforms.Resize(original_size)(variation_image)
#     # Rescale pixel values to [0, 1]
#     variation_image = (variation_image + 1) / 2.0
#     # Convert the tensor to a PIL image
#     variation_image = to_pil_image(variation_image)
#     variation_image.save(f"result_{idx + 1}.jpg")


import torch
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)
sd_pipe.safety_checker = lambda images, clip_input: (images, False)
sd_pipe = sd_pipe.to(device)

sd_pipe.enable_attention_slicing()
# sd_pipe.enable_sequential_cpu_offload()

im = Image.open("input_image_2.jpg")
tform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ]
)
inp = tform(im).to(device).unsqueeze(0)

out = sd_pipe(inp, guidance_scale=3)
for c, i in enumerate(out["images"]):
    i.save(f"result_{c}.jpg")
