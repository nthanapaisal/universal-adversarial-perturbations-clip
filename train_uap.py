import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip
import random 

IMAGE_DIR_FLICKR = "Flickr30k/Images"
IMAGE_DIR_COCO = "MSCOCO/"
TOTAL_IMAGES = 3000
TRAINING_SIZE = 2000
BATCH_SIZE = 100
TESTING_SIZE = 1000

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# get device of this machine
DEVICE = get_device()
print("Using device:", DEVICE)

def load_images(image_dir, preprocess):
    # load image from paths, convert ot RGB, preprocess before append to list
    paths = random.sample(list(Path(image_dir).glob("*.jpg")), TOTAL_IMAGES) 
    if not paths:
        raise ValueError(f"No .jpg images found in {image_dir}")

    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")  # img-> PIL Image
        imgs.append(preprocess(img))        #pre processing into tensor of shape [3, 224, 224]
        #tensor([
        #   [[ 0.12,  0.08, -0.03],
        #    [ 0.20,  0.15,  0.02],
        #    [ 0.30,  0.25,  0.10]],   # RED channel

        #   [[-0.10, -0.05,  0.00],
        #    [ 0.05,  0.10,  0.15],
        #    [ 0.20,  0.25,  0.30]],   # GREEN channel

        #   [[-0.50, -0.40, -0.30],
        #    [-0.20, -0.10,  0.00],
        #    [ 0.10,  0.20,  0.30]]    # BLUE channel
        # ])

    # turn list of tensors into tensor structure "batch tensor" with all img tensors in it.  -> [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
    return imgs, paths


@torch.no_grad() # do not track gradients inside this func
def encode_text(model, tokenizer, prompts):
    tokens = tokenizer(prompts).to(DEVICE) #tokenize strings and send to mps
    text_features = model.encode_text(tokens) #pass tokens to clip
    text_features = F.normalize(text_features, dim=-1)
    return text_features


@torch.no_grad()
def eval_sim(model, images, text_features):
    image_features = model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1)
    sims = image_features @ text_features.T #[N, D] * [D, M]
    best_scores, best_idx = sims.max(dim=1) # highest similarity, which text matched best
    return best_scores.mean().item(), best_idx # average similarity across all images, item() convert tensor to float

def eval_full_set(model, images_list, text_features, delta, batch_size):
    clean_scores_all = []
    adv_scores_all = []
    clean_idx_all = []
    adv_idx_all = []

    with torch.no_grad():
        for i in range(0, len(images_list), batch_size):
            batch_imgs = images_list[i:i + batch_size]
            batch = torch.stack(batch_imgs).to(DEVICE)

            # clean
            image_features = model.encode_image(batch)
            image_features = F.normalize(image_features, dim=-1)
            sims = image_features @ text_features.T
            clean_scores, clean_idx = sims.max(dim=1)

            # adversarial
            adv = torch.clamp(batch + delta, 0, 1)
            adv_features = model.encode_image(adv)
            adv_features = F.normalize(adv_features, dim=-1)
            adv_sims = adv_features @ text_features.T
            adv_scores, adv_idx = adv_sims.max(dim=1)

            clean_scores_all.append(clean_scores.cpu())
            adv_scores_all.append(adv_scores.cpu())
            clean_idx_all.append(clean_idx.cpu())
            adv_idx_all.append(adv_idx.cpu())

    clean_scores_all = torch.cat(clean_scores_all)
    adv_scores_all = torch.cat(adv_scores_all)
    clean_idx_all = torch.cat(clean_idx_all)
    adv_idx_all = torch.cat(adv_idx_all)

    clean_mean = clean_scores_all.mean().item()
    adv_mean = adv_scores_all.mean().item()

    changed = (clean_idx_all != adv_idx_all).sum().item()

    print("\nResults:")
    print(f"Clean similarity:  {clean_mean:.4f}")
    print(f"Attack similarity: {adv_mean:.4f}")
    print(f"Changed predictions: {changed}/{len(images_list)}")

    return clean_mean, adv_mean, clean_idx_all, adv_idx_all

def train_clipuap(eps, steps, lr, prompts):
    # get model: clip - resnet-50 backbone for img and transformer for txt, uses open ai weights
    # preprocess is function prepare raw image for clip model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50", pretrained="openai"
    )
    # need tokenizaer for text encoder: string -> token ids
    tokenizer = open_clip.get_tokenizer("RN50") # rn50 does not mean tokenizer for rn50 but for clip with backbone rn50

    # move all model weights to that device
    model = model.to(DEVICE)

    # tell model we are inferencing mode
    model.eval()

    # freeze model weights because .eval() does not stop gradients
    for p in model.parameters():
        p.requires_grad_(False)

    # embedding txt and normalize 
    text_features = encode_text(model, tokenizer, prompts)

    # load images
    images, paths = load_images(IMAGE_DIR_FLICKR, preprocess)

    # a trainable noise tensor that will be added to every image
    delta = torch.zeros((1, *images[0].shape), device=DEVICE, requires_grad=True) # grad tracks delta → adv → model → embeddings → loss

    for step in tqdm(range(steps)):
        for batch_idx in range(0, TRAINING_SIZE, BATCH_SIZE):
            print(f"Batch: {batch_idx}")
            batch_imgs = images[batch_idx: batch_idx + BATCH_SIZE]
            images_train = torch.stack(batch_imgs).to(DEVICE)
            #CREATE ADVERSARIAL IMAGE RANGE[0,1], forces every value in x to stay within [min, max]
            adv = torch.clamp(images_train + delta, 0, 1)  
            # reencode adversarial image
            image_features = model.encode_image(adv)
            image_features = F.normalize(image_features, dim=-1)
            # compute similarity
            sims = image_features @ text_features.T
            # best scores
            best_scores, _ = sims.max(dim=1)
            # take avg base score and make it loss (finding strong image-text matches) -trying to reduce CLIP’s confidence in the best matching text
            # average of the best image-text similarity for the batch
            # without txt we can do something like " image and original image loss (Example A: push embeddings away from original)"
            loss = best_scores.mean()
            # compute gradient respect to delta
            grad = torch.autograd.grad(loss, delta)[0]
            # update noise
            with torch.no_grad(): # Do not track gradients while manually editing delta, parameter update step
                delta -= lr * grad.sign() #So each pixel moves by a fixed small amount in the direction that changes the loss.
                delta.clamp_(-eps, eps) #limits the perturbation size becasue the noise might grow huge and become obvious.

            
            print(f"step {step} loss {loss.item():.4f}")


    ##### end of training noise #####
    print("\nEvaluating on training set...")
    train_images = images[:TRAINING_SIZE]
    eval_full_set(
        model=model,
        images_list=train_images,
        text_features=text_features,
        delta=delta,
        batch_size=BATCH_SIZE
    )

    print("\nEvaluating on test set...")
    test_images = images[TRAINING_SIZE: TRAINING_SIZE + TESTING_SIZE] #200 for both tests
    eval_full_set(
        model=model,
        images_list=test_images,
        text_features=text_features,
        delta=delta,
        batch_size=BATCH_SIZE
    )

def main():
    PROMPTS = [
    "a photo of a person",
    "a portrait of a person",
    "a person standing outdoors",
    "a person indoors",
    "a photo of an animal",
    "a dog or other animal",
    "a photo of a car or vehicle",
    "a vehicle on a road",
    "a street scene",
    "a city scene",
    "a natural landscape",
    "an outdoor scene",
    "an indoor scene",
    "a group of people",
        "a close-up object photo",
    ]

    EPS = 10 / 255.0 # after all iterations, each pixel is guaranteed to change by at most ~3%
    STEPS = 10
    LR = 1 / 255.0 # Each step changes pixels by exactly 1/255

    
    print("Testing CLIPUAP: FLICKR")
    train_clipuap(EPS, STEPS, LR, PROMPTS)
    
if __name__ == "__main__":
    main()