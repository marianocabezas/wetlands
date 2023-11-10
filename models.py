import math
from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel


class Classifier(BaseModel):
    def __init__(self):
        super().__init__()

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
                'weight': 0,
                'f': F.cross_entropy
            },

            {
                'name': 'acc',
                'weight': 0,
                'f': self._accuracy_loss
            },
            {
                'name': 'tpr',
                'weight': 1,
                'f': self._tpr_loss
            },
            {
                'name': 'tnr',
                'weight': 1,
                'f': self._tnr_loss
            }
        ]

    def _accuracy_loss(self, predicted, target):
        pred_y = torch.argmax(predicted, dim=1)
        return 1 - torch.mean((pred_y == target).to(dtype=torch.float32))

    def _tpr_loss(self, predicted, target):
        mask = target == 1
        pred_y = torch.argmax(predicted, dim=1)[mask]
        return 1 - torch.mean(pred_y.to(dtype=torch.float32))

    def _tnr_loss(self, predicted, target):
        mask = target == 0
        pred_y = torch.argmax(predicted, dim=1)[mask] == 0
        return 1 - torch.mean(pred_y.to(dtype=torch.float32))

    def forward(self, *inputs):
        return None


class Segmenter(BaseModel):
    def __init__(self, n_outputs):
        super().__init__()
        self.n_classes = n_outputs

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
                'weight': 0,
                'f': F.cross_entropy
            },

            {
                'name': 'dsc',
                'weight': 1,
                'f': self._dsc_loss
            },
            {
                'name': 'mIoU',
                'weight': 0,
                'f': self._mean_iou
            },
        ]

    def _dsc_loss(self, predicted, target):
        p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
        t = torch.flatten(target, start_dim=1).to(predicted.device)
        intersection = torch.stack([
            2 * torch.sum(
                (p == label).type_as(p) * (t == label).type_as(p),
                dim=1
            )
            for label in range(self.n_classes)
        ])
        sum_pred = torch.stack([
            torch.sum((p == label).type_as(p), dim=1)
            for label in range(self.n_classes)
        ])
        sum_target = torch.stack([
            torch.sum((t == label).type_as(p), dim=1)
            for label in range(self.n_classes)
        ])
        dsc_k = torch.mean(intersection / (sum_pred + sum_target), dim=0)
        dsc_k = dsc_k[torch.logical_not(torch.isnan(dsc_k))]
        if len(dsc_k) > 0:
            dsc = 1 - torch.mean(dsc_k)
        else:
            dsc = torch.mean(0. * p)

        return torch.clamp(dsc, 0., 1.)

    def _mean_iou(self, predicted, target):
        p = torch.flatten(torch.argmax(predicted, dim=1), start_dim=1)
        t = torch.flatten(target, start_dim=1).to(predicted.device)
        intersection = torch.stack([
            torch.sum(
                torch.logical_or(p == label, t == label).type_as(p), dim=1
            )
            for label in range(self.n_classes)
        ])
        union = torch.stack([
            torch.sum(
                torch.logical_or(p == label, t == label).type_as(p), dim=1
            )
            for label in range(self.n_classes)
        ])
        miou_k = torch.mean(intersection / union, dim=0)
        miou_k = miou_k[torch.logical_not(torch.isnan(miou_k))]
        if len(miou_k) > 0:
            miou = 1 - torch.mean(miou_k)
        else:
            miou = torch.mean(0. * p)

        return torch.clamp(miou, 0., 1.)

    def forward(self, *inputs):
        return None


class ConvNeXtTiny(Classifier):
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

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        self.cnext.to(self.device)
        return self.cnext(data)

    def target_layer(self):
        return self.cnext.features[3]


class ResNet18(Classifier):
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

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        self.resnet.to(self.device)
        return self.resnet(data)

    def target_layer(self):
        return self.resnet.layer3


class ResNet101(Classifier):
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
                self.resnet = models.resnet101(weights='IMAGENET1K_V1')
            except TypeError:
                self.resnet = models.resnet101(pretrained)
        else:
            self.resnet = models.resnet18()
        self.last_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.last_features, self.n_classes)

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        self.resnet.to(self.device)
        return self.resnet(data)

    def target_layer(self):
        return self.resnet.layer3


class ViT_B_16(Classifier):
    def __init__(
        self, image_size, patch_size, n_outputs, pretrained=False, lr=1e-3,
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
                #image_size=image_size,
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1
            )
        else:
            self.vit = models.vit_b_16(
                #image_size=image_size,
                weights=None
            )

        if patch_size != 16:
            self.vit.conv_proj = nn.Parameter(
                F.interpolate(
                    self.vit.conv_proj.weight,
                    size=patch_size,
                    mode='bicubic',
                    align_corners=True
                ),
                requires_grad=True,
            )

        pos_embedding = self.vit.encoder.pos_embedding
        n, seq_length, hidden_dim = pos_embedding.shape
        new_seq_length = (image_size // patch_size) ** 2 + 1
        if new_seq_length != seq_length:
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]

            # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))

            # (1, hidden_dim, seq_length)
            #                 |
            #                 v
            # (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(
                1, hidden_dim, seq_length_1d, seq_length_1d
            )
            new_seq_length_1d = image_size // patch_size

            # Perform interpolation.
            # (1, hidden_dim, seq_l_1d, seq_l_1d)
            #                 |
            #                 v
            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode='bicubic',
                align_corners=True,
            )

            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
            #                 |
            #                 v
            # (1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.reshape(
                1, hidden_dim, new_seq_length
            )

            # (1, hidden_dim, new_seq_length)
            #                 |
            #                 v
            # (1, new_seq_length, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
            new_pos_embedding = torch.cat(
                [pos_embedding_token, new_pos_embedding_img], dim=1
            )
            self.vit.encoder.pos_embedding = nn.Parameter(
                new_pos_embedding,
                requires_grad=True
            )

        if self.vit.image_size != image_size:
            self.vit.image_size = image_size

        if self.vit.patch_size != patch_size:
            self.vit.patch_size = patch_size

        self.last_features = self.vit.heads[0].in_features
        self.vit.heads[0] = nn.Linear(self.last_features, self.n_classes)

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        self.vit.to(self.device)
        return self.vit(data)

    def target_layer(self):
        return self.vit.conv_proj


class FCN_ResNet50(Segmenter):
    def __init__(
        self, n_inputs, n_outputs, pretrained=False, lr=1e-3,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        verbose=True
    ):
        super().__init__(n_outputs)
        # Init
        self.channels = n_inputs
        self.lr = lr
        self.device = device
        if pretrained:
            try:
                weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT
                self.fcn = models.segmentation.fcn_resnet50(weights=weights)
            except TypeError:
                self.fcn = models.segmentation.fcn_resnet50(pretrained)
        else:
            self.fcn = models.segmentation.fcn_resnet50
        if n_inputs > 3:
            conv_input = nn.Conv2d(
                n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # We assume that RGB channels will be the first 3
            conv_input.weight.data[:, :3, ...].copy_(
                self.fcn.backbone.conv1.weight.data
            )
            self.fcn.backbone.conv1 = conv_input
        elif n_inputs < 3:
            self.fcn.backbone.conv1 = nn.Conv2d(
                n_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.last_features = self.fcn.classifier[-1].in_channels
        self.fcn.classifier[-1] = nn.Conv2d(
            self.last_features, 2, kernel_size=1, stride=1
        )
        self.aux_last_features = self.fcn.aux_classifier[-1].in_channels
        self.fcn.aux_classifier[-1] = nn.Conv2d(
            self.aux_last_features, 2, kernel_size=1, stride=1
        )

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=self.lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        self.fcn.to(self.device)
        return self.fcn(data)['out']

    def target_layer(self):
        return self.fcn.backbone
