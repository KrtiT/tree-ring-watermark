import matplotlib
import matplotlib.pyplot as plt

from skimage.transform import resize

import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

import torch.nn as nn
import numpy as np

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *

##
# RISE Code from: https://github.com/eclique/RISE/blob/master/explanations.py
##
class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks#.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image

        stack = torch.mul(self.masks, x.data)
        # stack = torch.mul(self.masks.to('cpu'), x.data.to('cpu'))

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal

def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]

        # print(orig_image_w)
        orig_image_w.save("img1.png","PNG")
        
        ### test watermark
        # distortion
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)

        orig_image_no_w_auged.save("img2.png","PNG")
        orig_image_w_auged.save("img3.png","PNG")

        import torchvision
        from torchvision.models import efficientnet_b5
        net = efficientnet_b5(pretrained=True)
        orig_image_w = orig_image_w.resize((256,256))
        input_size = orig_image_w.size # (256,256)
        orig_image_w = torchvision.transforms.functional.pil_to_tensor(orig_image_w).float().unsqueeze(0)
        
        batch_size = 1
        N = 1
        s = 8
        p1 = 0.5

        rise = RISE(net, input_size, gpu_batch=1)
        rise.generate_masks(N, s, p1, savepath='masks.npy')
        sal = rise(orig_image_w)

        p, c = torch.topk(net(orig_image_w), k=3)
        p, c = p[0], c[0]
        sal = sal[c[0]]

        plt.figure()
        # sal = sal.numpy().transpose((1, 2, 0))[:,:,0]
        sal = np.expand_dims(sal, axis=-1)
        plt.imshow(sal)
        plt.savefig('img_heat.png')

        plt.figure()
        plt.imshow(orig_image_w.squeeze(0).numpy().transpose((1, 2, 0)).astype(np.uint8))
        plt.imshow(sal.astype(np.uint8), cmap='jet', alpha=0.5)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig('test.png')

        ##
        # given a threshold make the saliency/segmentation map
        ##
        # print(np.min(sal),np.max(sal))
        threshold = 0.8
        sal_mask = np.zeros(sal.shape)
        mask = sal > threshold
        sal_mask[mask] = 255
        sal_mask[~mask] = 0
        sal[mask] = 1
        sal[~mask] = 0
        sal_mask = sal_mask.astype(np.uint8)
        sal = sal.astype(np.uint8)
        plt.imshow(sal_mask, cmap='gray')
        plt.savefig('test2.png')

        ##
        # multiplying the saliency map threshold with the original image
        ##
        plt.figure()
        plt.imshow(sal*orig_image_w.squeeze(0).numpy().transpose((1, 2, 0)).astype(np.uint8))
        plt.savefig('test3.png')

        ##
        # multiplying the saliency map threshold with random noise
        ##
        plt.figure()
        noise = np.random.randint(low=0, high=255, size=sal.shape)
        plt.imshow(sal*noise.astype(np.uint8))
        plt.savefig('test4.png')

        ##
        # NOTE: Issue with warning
        # WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
        ##
        plt.figure()
        new_image = sal*np.random.randint(low=0, high=255, size=orig_image_w.squeeze(0).numpy().transpose((1, 2, 0)).shape) + orig_image_w.squeeze(0).numpy().transpose((1, 2, 0)).astype(np.uint8)
        plt.imshow(new_image)
        plt.savefig('test5.png')
        
        exit()

        # reverse img without watermarking
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse img with watermarking
        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)

        if args.reference_model is not None:
            sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0
            w_sim = 0

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric, 'w_no_sim': w_no_sim, 'w_sim': w_sim,
        })

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # log images when we use reference_model
                table.add_data(wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, current_prompt, no_w_metric, w_metric)
            else:
                table.add_data(None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric)

            clip_scores.append(w_no_sim)
            clip_scores_w.append(w_sim)

    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                   'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                   'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
    
    print(f'clip_score_mean: {mean(clip_scores)}')
    print(f'w_clip_score_mean: {mean(clip_scores_w)}')
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)