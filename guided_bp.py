import torch
from torch import nn
from utils import *


class GuidedBackPropagation(object):

    def __init__(self, net):
        self.net = net
        for (name, module) in self.net.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(self.backward_hook)
        self.net.eval()

    @classmethod
    def backward_hook(cls, module, grad_in, grad_out):
        """
        :param module:
        :param grad_in: tuple,长度为1
        :param grad_out: tuple,长度为1
        :return: tuple(new_grad_in,)
        """
        return torch.clamp(grad_in[0], min=0.0),

    def __call__(self, inputs, index=None):
        """
        :param inputs: [1,3,H,W]
        :param index: class_id
        :return:
        """
        inputs.requires_grad = True
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]

        target.backward()
        # print(inputs.grad, inputs.requires_grad)
        if inputs.is_cuda:
            return inputs.grad[0].cpu().data.numpy()
        else:
            return inputs.grad[0].data.numpy()  # [3,H,W]


if __name__ == '__main__':
    device = tc.device('cpu')
    model_name = 'resnet18'
    net = load_net(model_name)
    net = change_classifier(net, model_name, 3)
    net.load_state_dict(tc.load('./saved_models/{}.pth'.format(model_name), map_location=device))
    net.to(device).float()
    file_name_to_export = 'normal'

    # img = cv_imread('./data2/水肿/白永禄002.jpg')
    # img = cv_imread('./data2/萎缩/毕玉林003.jpg')
    img = cv_imread('./data2/正常/金芝芳00.jpg')
    img_ts = preprocess_img(img)

    # Guided backprop
    GBP = GuidedBackPropagation(net)
    # Get gradients
    guided_grads = GBP(img_ts, 1)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')