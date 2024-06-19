# Neural Style Transfer Report

## Introduction

Neural Style Transfer (NST) is a technique that involves blending the content of one image with the style of another using convolutional neural networks (CNNs). This technique is widely used in digital art and design, enabling artists to create unique images by merging different artistic styles with various contents. In this project, we implemented an NST algorithm using PyTorch, applying the style of famous artworks to various content images.

## Methodology

### Overview

The goal of this project is to transfer the style of a style image to a content image, resulting in a new image that maintains the content of the original content image but adopts the artistic style of the style image. We used a pre-trained VGG19 network for feature extraction and defined a custom model to compute content and style losses. The process involves the following steps:

1. Load and preprocess the content and style images.
2. Define a custom neural network model for style transfer.
3. Compute content and style losses.
4. Optimize the input image to minimize the combined loss.
5. Generate and save the stylized image.

### Image Preprocessing

We began by loading and preprocessing the images. The images were resized to a fixed size (512x512 for GPU and 128x128 for CPU) to ensure consistency in processing.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 128
transform = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])

def load_image(image_path, size=None):
    image = Image.open(image_path)
    if size is not None:
        image = transforms.Resize(size)(image)
    image = transform(image).unsqueeze(0)
    return image.to(device)
```

### Custom Neural Network Model

We used a pre-trained VGG19 network for feature extraction. The VGG19 model is known for its deep architecture and capability to capture complex patterns in images, making it ideal for style transfer tasks.

#### Normalization

Normalization was applied to the input images to match the pre-trained VGG19 network's expected input format. The normalization layer adjusted the mean and standard deviation of the input images.

```python
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std
```

#### Content and Style Losses

To compute the content and style losses, we defined two custom layers: `ContentLoss` and `StyleLoss`. The content loss measures the difference between the content image and the generated image in terms of high-level features, while the style loss measures the difference in the style of the images using Gram matrices.

```python
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = StyleTransfer.gram_matrix(self, target_feature).detach()

    def forward(self, input):
        G = StyleTransfer.gram_matrix(self, input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input
```

#### Gram Matrix

The Gram matrix is a mathematical representation of the style of an image. It captures the correlation between different features in the image.

```python
def gram_matrix(self, input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)
```

### Model Construction

The `StyleTransfer` class combines the normalization layer, content loss, and style loss into a single model. It extracts features from different layers of the VGG19 network to compute the losses.

```python
class StyleTransfer(nn.Module):
    def __init__(self, cnn, normalization_mean, normalization_std,
                 content_image, style_image, content_layers=None, style_layers=None):
        super(StyleTransfer, self).__init__()
        self.cnn = copy.deepcopy(cnn)
        self.normalization = Normalization(normalization_mean, normalization_std).to(device)

        if content_layers is None:
            content_layers = ['conv_4']
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.content_layers = content_layers
        self.style_layers = style_layers

        self.model, self.content_losses, self.style_losses = self.get_model_and_losses(content_image, style_image)

    def get_model_and_losses(self, content_image, style_image):
        content_losses = []
        style_losses = []
        model = nn.Sequential(self.normalization)
        i = 0

        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, content_losses, style_losses

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
```

### Style Transfer Optimization

We used the L-BFGS optimizer to iteratively update the input image to minimize the combined style and content losses. The input image was initialized as a clone of the content image, and the optimization process adjusted the pixel values to match the style of the style image while preserving the content.

```python
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(style_transfer, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
     
    optimizer = get_input_optimizer(input_img)
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            style_transfer.model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_transfer.style_losses:
                style_score += sl.loss
            for cl in style_transfer.content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            sl=style_score
            cl=content_score
            if run[0] % 50 == 0:
                print(f"run {run[0]}:")
                print(f'Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}')
                print()

            return loss

        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img
```

### Implementation Details

1. **Dataset Preparation**: The content and style images were stored in separate directories. We processed all combinations of content and style images to generate the output images.
2. **Model Execution**: For each pair of content and style images, we built a style transfer model and ran the optimization process to generate the stylized image.
3. **Saving Outputs**: The output images were saved in a specified directory.

```python
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

working_dir = "/path/to/working_dir"
content_dir = os.path.join(working_dir, "content")
style_dir = os.path.join(working_dir, "style")
output_dir = os.path.join(working_dir, "output")
os.makedirs(output_dir, exist_ok=True)

content_images = os.listdir(content_dir)
style_images = os.listdir(style_dir)

for content_image_name in content_images:
    for style_image_name in style_images:
        content_image_path = os.path.join(content

_dir, content_image_name)
        style_image_path = os.path.join(style_dir, style_image_name)
        output_image_name = f"{os.path.splitext(content_image_name)[0]}_{os.path.splitext(style_image_name)[0]}.jpg"
        print("Building model for content image ",os.path.splitext(content_image_name)[0]," Vs style image ",os.path.splitext(style_image_name)[0])
        content_image = load_image(content_image_path)
        style_image = load_image(style_image_path, get_image_size(content_image))

        style_transfer = StyleTransfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_image, style_image)

        input_image = content_image.clone()

        output_image = run_style_transfer(style_transfer, input_image)

        output_image_path = os.path.join(output_dir, output_image_name)
        output_image_pil = transforms.ToPILImage()(output_image.squeeze(0).cpu())
        output_image_pil.save(output_image_path)

        print(f"Saved stylized image: {output_image_path}")
```

### Results

The results of the style transfer process were visually inspected by comparing the content, style, and output images. The output images successfully blended the content of the content images with the style of the style images.

```python
for content_image_name in content_images:
    content_image_path = os.path.join(content_dir, content_image_name)
    content_image = load_image(content_image_path)
    
    plt.figure(figsize=(100, 5 * len(style_images)))
    
    for idx, style_image_name in enumerate(style_images, start=1):
        style_image_path = os.path.join(style_dir, style_image_name)
        style_image = load_image(style_image_path, get_image_size(content_image))
        output_image_name = f"{os.path.splitext(content_image_name)[0]}_{os.path.splitext(style_image_name)[0]}.jpg"
        output_image_path = os.path.join(output_dir, output_image_name)
        output_image = load_image(output_image_path)
        
        row_index = (idx - 1) * 3
        
        plt.subplot(len(style_images), 3, row_index + 1)
        imshow(content_image, title='Content Image')
        
        plt.subplot(len(style_images), 3, row_index + 2)
        imshow(style_image, title='Style Image')
        
        plt.subplot(len(style_images), 3, row_index + 3)
        imshow(output_image, title='Output Image')
    
    plt.tight_layout()
    plt.show()
```

### Challenges

1. **Computation Time**: The optimization process is computationally intensive and time-consuming, especially when processing high-resolution images.
2. **Hyperparameter Tuning**: Finding the right balance between style and content weights was crucial for achieving visually appealing results.
3. **Image Quality**: Maintaining high image quality while transferring the style effectively was challenging.

## Conclusion

Neural Style Transfer is a powerful technique for blending artistic styles with different content images. By leveraging the capabilities of deep neural networks, we can create visually stunning images that combine the content of one image with the style of another. Despite the challenges in computation time and hyperparameter tuning, the results demonstrate the effectiveness of the NST algorithm in generating unique and artistic images. This project showcases the potential of deep learning in the field of digital art and design.