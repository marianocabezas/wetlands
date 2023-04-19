import time
from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel, ResConv3dBlock, ViTEncoder
from base import Autoencoder, AttentionAutoencoder, DualAttentionAutoencoder
from utils import time_to_string
from criteria import gendsc_loss, similarity_loss, grad_loss, accuracy
from criteria import tp_binary_loss, tn_binary_loss, dsc_binary_loss


def norm_f(n_f):
    return nn.GroupNorm(n_f // 4, n_f)


def print_batch(pi, n_patches, i, n_cases, t_in, t_case_in):
    init_c = '\033[38;5;238m'
    percent = 25 * (pi + 1) // n_patches
    progress_s = ''.join(['█'] * percent)
    remainder_s = ''.join([' '] * (25 - percent))

    t_out = time.time() - t_in
    t_case_out = time.time() - t_case_in
    time_s = time_to_string(t_out)

    t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
    eta_s = time_to_string(t_eta)
    pre_s = '{:}Case {:03d}/{:03d} ({:03d}/{:03d} - {:06.2f}%) [{:}{:}]' \
            ' {:} ETA: {:}'
    batch_s = pre_s.format(
        init_c, i + 1, n_cases, pi + 1, n_patches, 100 * (pi + 1) / n_patches,
        progress_s, remainder_s, time_s, eta_s + '\033[0m'
    )
    print('\033[K', end='', flush=True)
    print(batch_s, end='\r', flush=True)


class ConvNeXtTiny(BaseModel):
    def __init__(
        self, n_outputs, pretrained=False, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__()
        # Init
        self.n_classes = n_outputs
        self.lr = lr
        self.device = device
        if pretrained:
            try:
                self.cnext = models.convnext_tiny(weights='IMAGENET1K_V1')
            except TypeError:
                self.cnext = models.convnext_tiny(pretrained)
        else:
            self.cnext = models.convnext_tiny()
        self.last_features = self.cnext.classifier[-1].in_features
        self.cnext.classifier[-1] = nn.Linear(
            self.last_features, self.n_classes
        )

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': F.cross_entropy
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': F.cross_entropy
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=self.lr)

    def gram_matrix(self, data):
        # data = self.cnext.features[:3](data)
        data = self.cnext.features[:2](data)
        flat_data = torch.flatten(data, 2)
        G = torch.bmm(flat_data, flat_data.transpose(1, 2))
        norm = data.numel() / len(data)
        return G / norm

    def tokenize(self, data):
        data = self.cnext.features(data)
        return data.flatten(2).permute(0, 2, 1)

    def prelogits(self, data):
        return self.cnext.features(data).flatten(1)

    def features(self, data):
        data = self.cnext.features(data)
        return data.flatten(1)

    def forward(self, data):
        self.cnext.to(self.device)
        return self.cnext(data)


class ResNet18(BaseModel):
    def __init__(
        self, n_outputs, pretrained=False, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__()
        # Init
        self.n_classes = n_outputs
        self.lr = lr
        self.device = device
        if pretrained:
            try:
                self.resnet = models.resnet18(weights='IMAGENET1K_V1')
            except TypeError:
                self.resnet = models.resnet18(pretrained)
        else:
            self.resnet = models.resnet18()
        self.last_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.last_features, self.n_classes)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': F.cross_entropy
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': F.cross_entropy
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=self.lr)

    def gram_matrix(self, data):
        data = self.resnet.conv1(data)
        data = self.resnet.bn1(data)
        data = self.resnet.relu(data)
        data = self.resnet.maxpool(data)
        data = self.resnet.layer1(data)
        data = self.resnet.layer2(data)
        # data = self.resnet.layer3(data)
        flat_data = torch.flatten(data, 2)
        G = torch.bmm(flat_data, flat_data.transpose(1, 2))
        norm = data.numel() / len(data)
        return G / norm

    def forward(self, data):
        self.resnet.to(self.device)
        return self.resnet(data)

    def _feature_seq(self, data):
        data = self.resnet.conv1(data)
        data = self.resnet.bn1(data)
        data = self.resnet.relu(data)
        data = self.resnet.maxpool(data)
        data = self.resnet.layer1(data)
        data = self.resnet.layer2(data)
        data = self.resnet.layer3(data)
        return self.resnet.layer4(data)

    def tokenize(self, data):
        data = self._feature_seq(data)
        return data.flatten(2).permute(0, 2, 1)

    def prelogits(self, data):
        data = self._feature_seq(data)
        return self.resnet.avgpool(data)

    def features(self, data):
        data = self.resnet.conv1(data)
        data = self.resnet.bn1(data)
        data = self.resnet.relu(data)
        flat_1 = data.flatten(1)
        data = self.resnet.maxpool(data)
        data = self.resnet.layer1(data)
        flat_2 = data.flatten(1)
        data = self.resnet.layer2(data)
        flat_3 = data.flatten(1)
        data = self.resnet.layer3(data)
        flat_4 = data.flatten(1)
        data = self.resnet.layer4(data)
        flat_5 = data.flatten(1)
        return torch.cat([flat_1, flat_2, flat_3, flat_4, flat_5], dim=1)


class ViT_B_16(BaseModel):
    def __init__(
        self, image_size, patch_size,
        n_outputs, pretrained=False, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__()
        # Init
        self.n_classes = n_outputs
        self.lr = lr
        self.device = device
        if pretrained:
            self.vit = models.vit_b_16(
                image_size=image_size, patch_size=patch_size,
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1
            )
        else:
            self.vit = models.vit_b_16(
                image_size=image_size, patch_size=patch_size,
                weights=None
            )
        self.last_features = self.vit.heads[0].in_features
        self.vit.heads[0] = nn.Linear(self.last_features, self.n_classes)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': F.cross_entropy
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': F.cross_entropy
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=self.lr)

    def gram_matrix(self, data):
        data = self.vit._process_input(data)
        n = data.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        data = torch.cat([batch_class_token, data], dim=1)
        data = self.vit.encoder(data)

        flat_data = data[:, 1:].unsqueeze(1)
        G = torch.bmm(flat_data, flat_data.transpose(1, 2))
        norm = data.numel() / len(data)
        return G / norm

    def forward(self, data):
        self.vit.to(self.device)
        return self.vit(data)


def vitb_cifar(n_outputs, lr=1e-3, pretrained=False):
    return ViT_B_16(32, 2, n_outputs, pretrained, lr=lr)


def vitb_imagenet(n_outputs, lr=1e-3, pretrained=False):
    return ViT_B_16(64, 4, n_outputs, pretrained, lr=lr)


class SimpleUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.segmenter = nn.Sequential(
            Autoencoder(
                self.conv_filters, device, n_images, block=ResConv3dBlock,
                norm=norm_f
            ),
            nn.Conv3d(self.conv_filters[0], 1, 1)
        )
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'pdsc',
                'weight': 1,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            },
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, data):

        return torch.sigmoid(self.segmenter(data))


class XentrUNet(SimpleUNet):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__(conv_filters, device, n_images, dropout)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            },
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )


class SimpleResNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = True
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.extractor = Autoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.extractor.to(device)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_filters[-1], self.conv_filters[-1] // 2),
            nn.ReLU(),
            norm_f(self.conv_filters[-1] // 2),
            # nn.Linear(self.conv_filters[-1] // 2, self.conv_filters[-1] // 4),
            # nn.ReLU(),
            # norm_f(self.conv_filters[-1] // 4),
            # nn.Linear(self.conv_filters[-1] // 4, 1)
            nn.Linear(self.conv_filters[-1] // 2, 1)
        )
        self.classifier.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                # 'f': lambda p, t: focal_loss(
                    p, t.type_as(p).to(p.device)
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 0,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device)
                )
            },
            {
                'name': 'fn',
                'weight': 0.5,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0.5,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
            {
                'name': 'acc',
                'weight': 0,
                'f': lambda p, t: 1 - accuracy(
                    (p > 0.5).type_as(p), t.type_as(p)
                )
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-4)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=1e-4)

    def forward(self, data):
        _, features = self.extractor.encode(data)
        # final_features = torch.mean(features.flatten(2), dim=2)
        final_features = torch.max(features.flatten(2), dim=2)[0]
        logits = self.classifier(final_features)
        return torch.sigmoid(logits)

    def inference(self, data, nonbatched=False):
        return super().inference(data, nonbatched=nonbatched)

    def embeddings(self, data, nonbatched=False):
        with torch.no_grad():
            if isinstance(data, list) or isinstance(data, tuple):
                x_cuda = tuple(
                    torch.from_numpy(x_i).to(self.device)
                    for x_i in data
                )
                if nonbatched:
                    x_cuda = tuple(
                        x_i.unsqueeze(0) for x_i in x_cuda
                    )
                _, features = self.extractor.encode(*x_cuda)
            else:
                x_cuda = torch.from_numpy(data).to(self.device)
                if nonbatched:
                    x_cuda = x_cuda.unsqueeze(0)
                _, features = self.extractor.encode(x_cuda)
            torch.cuda.empty_cache()

        embeddings = torch.max(features.flatten(2), dim=2)[0]

        return embeddings.cpu().numpy()


class CTResNet(BaseModel):
    def __init__(
        self,
        conv_filters=None, n_outputs=7, n_images=1, verbose=0, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.init = True
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.extractor = Autoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.extractor.to(device)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_filters[-1], self.conv_filters[-1] // 2),
            nn.ReLU(),
            norm_f(self.conv_filters[-1] // 2),
            # nn.Linear(self.conv_filters[-1] // 2, self.conv_filters[-1] // 4),
            # nn.ReLU(),
            # norm_f(self.conv_filters[-1] // 4),
            # nn.Linear(self.conv_filters[-1] // 4, 1)
            nn.Linear(self.conv_filters[-1] // 2, n_outputs)
        )
        self.classifier.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(
                    p, t.type_as(p).to(p.device)
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy_with_logits(
                    p, t.type_as(p).to(p.device)
                )
            },
            {
                'name': 'acc',
                'weight': 0,
                'f': lambda p, t: 1 - accuracy(
                    (torch.sigmoid(p) > 0.5).type_as(p), t.type_as(p)
                )
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=1e-4)

    def forward(self, data):
        _, features = self.extractor.encode(data)
        # final_features = torch.mean(features.flatten(2), dim=2)
        final_features = torch.max(features.flatten(2), dim=2)[0]
        return self.classifier(final_features)

    def inference(self, data, nonbatched=False, task=None):
        with torch.no_grad():
            if isinstance(data, list) or isinstance(data, tuple):
                x_cuda = tuple(
                    torch.from_numpy(x_i).to(self.device)
                    for x_i in data
                )
                if nonbatched:
                    x_cuda = tuple(
                        x_i.unsqueeze(0) for x_i in x_cuda
                    )

                logits = self(*x_cuda)
            else:
                x_cuda = torch.from_numpy(data).to(self.device)
                if nonbatched:
                    x_cuda = x_cuda.unsqueeze(0)
                logits = self(x_cuda)
            torch.cuda.empty_cache()
            output = torch.sigmoid(logits)

            if nonbatched:
                np_output = output[0, 0].cpu().numpy()
            else:
                np_output = output.cpu().numpy()

        return np_output

    def embeddings(self, data, nonbatched=False):
        with torch.no_grad():
            if isinstance(data, list) or isinstance(data, tuple):
                x_cuda = tuple(
                    torch.from_numpy(x_i).to(self.device)
                    for x_i in data
                )
                if nonbatched:
                    x_cuda = tuple(
                        x_i.unsqueeze(0) for x_i in x_cuda
                    )
                _, features = self.extractor.encode(*x_cuda)
            else:
                x_cuda = torch.from_numpy(data).to(self.device)
                if nonbatched:
                    x_cuda = x_cuda.unsqueeze(0)
                _, features = self.extractor.encode(x_cuda)
            torch.cuda.empty_cache()

        embeddings = torch.max(features.flatten(2), dim=2)[0]

        return embeddings.cpu().numpy()


class AttentionUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.segmenter = nn.Sequential(
            AttentionAutoencoder(
                self.conv_filters, device, n_images, block=ResConv3dBlock,
                norm=norm_f
            ),
            ResConv3dBlock(
                self.conv_filters[0], self.conv_filters[0], 1,
                norm=norm_f
            ),
            nn.Conv3d(self.conv_filters[0], 1, 1)
        )
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'grad',
                'weight': 1,
                'f': lambda p, t: grad_loss(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xent',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            },
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, data):

        return torch.sigmoid(self.segmenter(data))


class DualHeadedUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=3,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [16, 32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.ae = DualAttentionAutoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.ae.to(device)
        self.segmenter = nn.Conv3d(self.conv_filters[0], 1, 1)
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'pdsc',
                'weight': 1,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'xentropy',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            }
        ]

        self.val_functions = [
            {
                'name': 'xentropy',
                'weight': 0,
                'f': lambda p, t: F.binary_cross_entropy(
                    p, t.type_as(p).to(p.device),
                )
            },
            {
                'name': 'pdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(p, t, w_bg=0, w_fg=1)
            },
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(p, t)
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(p, t)
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(p, t)
            },
        ]

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def reset_optimiser(self, model_params=None):
        super().reset_optimiser(model_params)
        if model_params is None:
            model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, source, target):
        features = self.ae(source, target)
        segmentation = torch.sigmoid(self.segmenter(features))
        return segmentation


class LongitudinalEncoder(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            n_images=1,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if conv_filters is None:
            self.conv_filters = [16, 32, 64, 128, 256, 512]
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>
        self.ae = Autoencoder(
            self.conv_filters, device, n_images, block=ResConv3dBlock,
            norm=norm_f
        )
        self.ae.to(device)
        self.final_source = ResConv3dBlock(
            self.conv_filters[0], 1, 1, nn.Identity, nn.Identity
        )
        self.final_source.to(device)
        self.final_target = ResConv3dBlock(
            self.conv_filters[0], 1, 1, nn.Identity, nn.Identity
        )
        self.final_target.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'bl',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(
                    p[0], t[0],
                )
            },
            {
                'name': 'fu',
                'weight': 1,
                'f': lambda p, t: F.mse_loss(
                    p[1], t[1],
                )
            },
            {
                'name': 'sim',
                'weight': 1,
                'f': lambda p, t: similarity_loss(p[2])
            },
        ]
        self.val_functions = self.train_functions

        self.update_logs()

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, source, target):
        source_out, source_feat = self.ae(source, keepfeat=True)
        target_out, target_feat = self.ae(target, keepfeat=True)

        source_out = self.final_source(source_out)
        target_out = self.final_source(target_out)

        feat = list(zip(source_feat, target_feat))

        return source_out, target_out, feat
