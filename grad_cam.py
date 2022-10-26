import torch
import torch.nn as nn
import numpy as np
#register_forward_hook must be used before forward
#1
class SaveValues():
    def __init__(self, layer):
        self.model  = None
        self.input  = None
        self.output = None
        self.grad_input  = None
        self.grad_output = None
        self.forward_hook  = layer.register_forward_hook(self.hook_fn_act)
        self.backward_hook = layer.register_full_backward_hook(self.hook_fn_grad)
#hook_fn_act must include three parameters modeule ,input,output
    def hook_fn_act(self, module, input, output):
        self.model  = module
        self.input  = input[0]
        self.output = output
    def hook_fn_grad(self, module, grad_input, grad_output):
        self.grad_input  = grad_input[0]
        self.grad_output = grad_output[0]
    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(2, 5)
        self.l2 = nn.Linear(5, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

l1loss = nn.L1Loss()
model  = Net()
value  = SaveValues(model.l2)
gt = torch.ones((10,), dtype=torch.float32, requires_grad=False)
x  = torch.ones((2,), dtype=torch.float32, requires_grad=False)

y = model(x)
loss  = l1loss(y, gt)
loss.backward()
x += 1.2
value.remove()


#2
#这部分是用来保存梯度值
class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model):
        self.model = model
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)  # Save the convolution output on that layer

        x = self.model.layer2(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        x = self.model.layer3(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        x = self.model.layer4(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        return conv_output, x
    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return conv_output, x


#这部分是用来可视化生成的3D的关注度
class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_layer, target_class=None):

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads
        self.model.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients,gradients
        # layer0:(1,512,7,7), layer1:(1,256,14,14), layer2:(1,128,28,28), layer3:(1,64,56,56),
        # 与后面conv_output是反的，因此需要逆序处理
        guided_gradients = self.extractor.gradients[-1 - target_layer].data.numpy()[0]

        # Get convolution outputs
        # layer0.shape:(64,56,56) layer1:(128,28,28) layer2:(256,14,14) layer3:(512,7,7)
        target = conv_output[target_layer].data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)

        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam_resize = Image.fromarray(cam).resize((input_image.shape[2],
                                                  input_image.shape[3]), Image.ANTIALIAS)
        cam = np.uint8(cam_resize) / 255

        return cam

