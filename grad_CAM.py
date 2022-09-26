from utils import *
from cam_utils import *
import ipdb


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name=None):
        self.net = net
        if layer_name is None:
            self.layer_name = get_last_conv_name(self.net)
        else:
            self.layer_name = layer_name
        #print(self.layer_name)
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        #ipdb.set_trace()

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                  input_grad[1]: weight
                                  input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index=None):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if not tc.is_tensor(output):
            output = output[0]
        if index is None:
            index = int(tc.argmax(output, dim=1).cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam, index


if __name__ == '__main__':
    device = tc.device('cpu')
    net = load_net('vgg16')
    net = change_classifier(net, 'vgg16', 3)
    net.load_state_dict(tc.load('./saved_models/vgg16.pth', map_location=device))
    net.to(device).float()
    img = cv_imread('./data/test0408/1/cao-xilong_001.jpg')
    # img = cv_imread('./data2/水肿/白永禄002.jpg')
    # img = cv_imread('./data2/萎缩/毕玉林003.jpg')
    # img = cv_imread('./data2/正常/金芝芳002.jpg')
    raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ts = preprocess_img(img)

    grad_cam = GradCAM(net, get_last_conv_name(net))
    cam = grad_cam(img_ts, 1)
    grad_cam.remove_handlers()
    print(cam)
    cam = np.uint8(cam * 255)
    height, width, _ = raw.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    result = heatmap * 0.3 + raw * 0.5
    Image.fromarray(np.uint8(result)).save('./grad_CAM/vgg16/{}-{}-{}.jpg'.format('水肿', '萎缩', '白永禄002'))