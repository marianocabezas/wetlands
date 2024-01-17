# basic python packages
import os
import time
from functools import  partial
from time import strftime

# data analysis packages
import numpy as np
from skimage import io as skio
from scipy.special import expit, softmax

# autograd packages
import torch
from torch.utils.data import DataLoader

# attribution methods
from captum.attr import IntegratedGradients, InputXGradient, DeepLift
from captum.attr import Deconvolution, Occlusion
from captum.attr import GuidedBackprop, GuidedGradCam, LayerGradCam

# repository functions
from utils import color_codes, normalise, time_to_string


def attribution(x, attr_m, *args, **kwargs):
    """
    Wrapper that contains the attribution method and a set of steps to convert
    the map into a pseudo-heatmap of the same shape as the input image.

    :param x: Original input image.
    :param attr_m: Function for the attribution method
    :param args: Additional arguments in a list format.
    :param kwargs: Additional arguments in a dictionary format.
    :return:
    """
    attr = torch.sigmoid(
        attr_m.attribute(
            x, target=1, *args, **kwargs
        ) / 1e-2
    )
    if x.shape != attr.shape:
        attr = F.interpolate(attr, size=x.size()[2:], mode='bilinear')
    attr_map = attr.squeeze().detach().cpu().numpy()
    return (attr_map * 255).astype(np.uint8)


def run_segmentation_experiments(
    master_seed, network_name, display_name, experiment_name, network_f,
    training_set, validation_set, testing_data, weight_path, maps_path,
    classes=None, patch_size=256, epochs=10, patience=5, n_seeds=30,
    n_inputs=3, n_classes=2, train_batch=20, test_batch=50, verbose=1
):
    # Init
    testing_mosaics, testing_masks = testing_data

    # Choosing random runs.
    np.random.seed(master_seed)
    seeds = np.random.randint(0, 100000, n_seeds)
    c = color_codes()

    dsc_list = []
    class_dsc_list = []
    # Main loop to run each independent random experiment.
    for test_n, seed in enumerate(seeds):
        acc = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

        # The network will only be instantiated with the number of output
        # classes. Therefore, networks that need extra parameters (like ViT)
        # will need to be passed as a partial function.
        net = network_f(n_inputs=n_inputs, n_outputs=n_classes)

        # This is a leftover from legacy code. If init is set to True (the
        # default option), a first validation epoch will be run to determine
        # the loss before training.
        net.init = False

        # The number of parameters is only captured for debugging and printing.
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

        if verbose > 1:
            print(
                '{:}[{:}] {:}Starting experiment '
                '{:}(seed {:05d} - {:} {:}[{:,} parameters]{:})'
                '{:} [{:02d}/{:02d}] {:}for {:} segmentation{:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, c['b'] + display_name,
                    c['nc'], n_param, c['y'],
                    c['nc'] + c['c'], test_n + 1, len(seeds),
                    c['nc'] + c['g'],
                    c['b'] + experiment_name + c['nc'] + c['g'], c['nc']
                )
            )

        training_loader = DataLoader(
            training_set, train_batch, True
        )
        validation_loader = DataLoader(
            validation_set, test_batch
        )
        model_path = os.path.join(
            weight_path, '{:}-balanced_s{:05d}_p{:03d}.pt'.format(
                network_name, seed, patch_size
            )
        )

        # For efficiency, we only run the code once. If the weights are
        # stored on disk, we do not need to train again.
        try:
            net.load_model(model_path)
        except IOError:
            net.train()
            print(''.join([' '] * 200), end='\r')
            net.fit(
                training_loader, validation_loader,
                epochs=epochs, patience=patience
            )
            net.save_model(model_path)

        if verbose > 2:
            print(''.join([' '] * 200), end='\r')
            print(
                '{:}[{:}] {:}Testing {:}(seed {:05d}){:} [{:02d}/{:02d}] '
                '{:}for {:} segmentation <{:03d} samples>{:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, c['nc'] + c['c'],
                    test_n + 1, len(seeds),
                    c['nc'] + c['g'],
                    c['b'] + experiment_name + c['nc'] + c['g'],
                    len(training_set), c['nc']
                )
            )

        # Metric evaluation.
        net.eval()
        with torch.no_grad():
            mosaic_dsc = []
            mosaic_class_dsc = []
            # Intermediate buffers for class metrics.
            for input_mosaic, mask_i in zip(testing_mosaics, testing_masks):
                pred_map = net.patch_inference(
                    normalise(input_mosaic).astype(np.float32),
                    patch_size, test_batch
                )

                pred_y = np.argmax(pred_map, axis=0).astype(np.uint8)
                y = mask_i.astype(np.uint8)
                intersection = np.stack([
                    2 * np.sum(np.logical_and(pred_y == lab, y == lab))
                    for lab in range(n_classes)
                ])
                card_pred_y = np.stack([
                    np.sum(pred_y == lab) for lab in range(n_classes)
                ])
                card_y = np.stack([
                    np.sum(y == lab) for lab in range(n_classes)
                ])
                dsc_k = intersection / (card_pred_y + card_y)
                dsc = np.nanmean(dsc_k)
                mosaic_dsc.append(dsc)
                mosaic_class_dsc.append(dsc_k.tolist())

                for i, map_i in enumerate(softmax(pred_map, axis=0)):
                    map_path = os.path.join(
                        maps_path,
                        '{:}-balanced_s{:05d}_map_{:02d}.png'.format(
                            network_name, seed, i
                        )
                    )
                    final_map = (255 * map_i).astype(np.uint8)
                    skio.imsave(map_path, final_map.astype(np.uint8))
                map_path = os.path.join(
                    maps_path, '{:}-balanced_s{:05d}_masks.png'.format(
                        network_name, seed
                    )
                )
                final_map = (255 * (pred_y / (n_classes - 1))).astype(np.uint8)
                skio.imsave(map_path, final_map.astype(np.uint8))

                dsc = np.nanmean(mosaic_dsc, axis=0)
                class_dsc = np.nanmean(mosaic_class_dsc, axis=0)

        bold_green = c['nc'] + c['g'] + c['b']
        if verbose > 2:
            print(''.join([' '] * 200), end='\r')
            print(
                '{:}[{:}] {:}DSC{:} (seed {:05d}){:} [{:02d}/{:02d}] {:}'
                '{:5.3f}{:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, c['nc'] + c['c'],
                    test_n + 1, len(seeds),
                    c['nc'] + c['b'], dsc, c['nc']
                )
            )

            class_dsc_s = ', '.join([
                '{:} {:5.3f}'.format(k, dsc_k)
                for k, dsc_k in zip(classes, class_dsc)
            ])
            print(
                '{:}[{:}] {:}Class DSC{:} '
                '(seed {:05d}){:} [{:02d}/{:02d}] {:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, c['nc'] + c['c'],
                    test_n + 1, len(seeds),
                    c['nc'] + c['b'] + class_dsc_s + c['nc']
                )
            )
        elif verbose > 1:
            print(''.join([' '] * 200), end='\r')
            print(
                '{:}Seed {:05d} {:} [{:,} parameters] '
                '{:}[{:02d}/{:02d}] {:} {:5.3f}{:}'.format(
                    c['y'], seed, c['b'] + display_name + c['nc'], n_param,
                    c['c'], test_n + 1, len(seeds),
                    bold_green + experiment_name + c['nc'] + c['b'],
                    dsc, c['nc']
                )
            )
        elif verbose > 0:
            print(''.join([' '] * 200), end='\r')
            print(
                '{:}Seed {:05d} {:} [{:,} parameters] '
                '{:}[{:02d}/{:02d}] {:} {:5.3f}{:}'.format(
                    c['y'], seed, c['b'] + display_name + c['nc'], n_param,
                    c['c'], test_n + 1, len(seeds),
                    bold_green + experiment_name + c['nc'] + c['b'],
                    dsc, c['nc']
                ), end='\r'
            )
        net = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        dsc_list.append(mosaic_dsc)
        class_dsc_list.append(mosaic_class_dsc)

    # Metrics for all the runs.
    if verbose > 0:
        print(''.join([' '] * 200), end='\r')
        print(
            '{:}[{:}] {:} Mean DSC{:} {:5.3f}{:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"),
                c['nc'] + c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                c['nc'] + c['b'], np.nanmean(dsc_list), c['nc']
            )
        )
        class_dsc_s = ', '.join([
            '{:} {:5.3f}'.format(k, dsc_k)
            for k, dsc_k in zip(
                classes, np.nanmean(class_dsc_list, axis=(0, 1))
            )
        ])
        print(
            '{:}[{:}] {:} Mean class DSC {:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"),
                c['nc'] + c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                c['nc'] + c['b'] + class_dsc_s + c['nc']
            )
        )

    return dsc_list, class_dsc_list


def run_classification_experiments(
    master_seed, network_name, display_name, name, network,
    training_set, testing_set, weight_path, classes=None,
    patch_size=256, epochs=10, patience=5, n_seeds=30, n_classes=2,
    train_batch=20, test_batch=50, verbose=1
):
    # Choosing random runs.
    np.random.seed(master_seed)
    seeds = np.random.randint(0, 100000, n_seeds)
    c = color_codes()
    if verbose > 1:
        print(
            '{:}[{:}] {:}Creating {:}testing{:} dataset '
            'for {:} classification{:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                c['nc'] + c['y'], c['nc'] + c['g'],
                c['b'] + name + c['nc'] + c['g'], c['nc']
            )
        )

    acc_list = []
    class_list = []
    # Main loop to run each independent random experiment.
    for test_n, seed in enumerate(seeds):
        acc = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

        # The network will only be instantiated with the number of output
        # classes. Therefore, networks that need extra parameters (like ViT)
        # will need to be passed as a partial function.
        net = network(n_outputs=n_classes)

        # This is a leftover from legacy code. If init is set to True (the
        # default option), a first validation epoch will be run to determine
        # the loss before training.
        net.init = False

        # The number of parameters is only captured for debugging and printing.
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

        if verbose > 1:
            print(
                '{:}[{:}] {:}Starting experiment '
                '{:}(seed {:05d} - {:} {:}[{:,} parameters]{:})'
                '{:} [{:02d}/{:02d}] {:}for {:} classification{:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, c['b'] + network_name,
                    c['nc'], n_param, c['y'],
                    c['nc'] + c['c'], test_n + 1, len(seeds),
                    c['nc'] + c['g'], c['b'] + name + c['nc'] + c['g'], c['nc']
                )
            )

        training_loader = DataLoader(
            training_set, train_batch, True
        )
        validation_loader = DataLoader(
            testing_set, test_batch
        )
        model_path = os.path.join(
            weight_path,
            '{:}-balanced_s{:05d}_p{:03d}.pt'.format(
                network_name, seed, patch_size
            )
        )

        # For efficiency, we only run the code once. If the weights are
        # stored on disk, we do not need to train again.
        try:
            net.load_model(model_path)
        except IOError:
            net.train()
            print(''.join([' '] * 200), end='\r')
            net.fit(
                training_loader, validation_loader,
                epochs=epochs, patience=patience
            )
            net.save_model(model_path)

        if verbose > 2:
            print(
                '{:}[{:}] {:}Testing {:}(seed {:05d} - {:}){:} '
                '[{:02d}/{:02d}] {:}for {:} classification '
                '<{:03d} samples>{:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, network_name,
                    c['clr'] + c['c'], test_n + 1, len(seeds),
                    c['nc'] + c['g'], c['b'] + 'rumex' + c['nc'] + c['g'],
                    len(training_set), c['nc']
                )
            )

        # Metric evaluation.
        net.eval()
        with torch.no_grad():
            # Intermediate buffers for class metrics.
            counts = np.zeros(n_classes)
            correct = np.zeros(n_classes)
            for i, (x, y) in enumerate(DataLoader(testing_set, test_batch)):
                pred_y = np.argmax(
                    net.inference(x.numpy()), axis=1
                ).astype(np.uint8)
                y = y.numpy().astype(np.uint8)
                acc += np.sum(y == pred_y) / len(testing_set)
                for k in range(n_classes):
                    k_mask = y == k
                    counts[k] += np.sum(k_mask)
                    correct[k] += np.sum(y[k_mask] == pred_y[k_mask])
            class_acc = [
                corr_k / num_k for corr_k, num_k in zip(correct, counts)
            ]
            acc_list.append(acc)
            class_list.append(class_acc)

        net = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if verbose > 1:
            print(''.join([' '] * 200), end='\r')
            print(
                '{:}[{:}] {:}Accuracy{:} (seed {:05d}){:} [{:02d}/{:02d}] {:}'
                '{:5.3f}{:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, c['nc'] + c['c'],
                    test_n + 1, len(seeds),
                    c['nc'] + c['b'], acc, c['nc']
                )
            )
            class_acc_s = ', '.join([
                'Class {:}: {:5.3f}'.format(k, acc_k)
                for k, acc_k in enumerate(class_acc)
            ])
            print(
                '{:}[{:}] {:}Class accuracies{:} (seed {:05d}){:} '
                '[{:02d}/{:02d}] {:}'.format(
                    c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['g'],
                    c['nc'] + c['y'], seed, c['nc'] + c['c'],
                    test_n + 1, len(seeds),
                    c['nc'] + c['b'] + class_acc_s + c['nc']
                )
            )
        elif verbose > 0:
            print(''.join([' '] * 200), end='\r')
            print(
                '{:}Seed {:05d} {:} [{:,} parameters] '
                '{:}[{:02d}/{:02d}] {:} {:5.3f}{:}'.format(
                    c['y'], seed, c['b'] + network_name + c['nc'], n_param,
                    c['c'], test_n + 1, len(seeds),
                    c['nc'] + c['g'] + c['b'] + name + c['nc'] + c['b'],
                    acc, c['nc']
                ), end='\r'
            )

    # Metrics for all the runs.
    if verbose > 0:
        print(''.join([' '] * 200), end='\r')
        print(
            '{:}[{:}] {:} Mean accuracy{:} {:5.3f}{:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"),
                c['nc'] + c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                c['nc'] + c['b'], np.nanmean(acc_list), c['nc']
            )
        )
        class_acc_s = ', '.join([
            '{:} {:5.3f}'.format(k, acc_k)
            for k, acc_k in zip(
                classes, np.nanmean(class_list, axis=0)
            )
        ])
        print(
            '{:}[{:}] {:} Mean class accuracy {:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"),
                c['nc'] + c['y'] + c['b'] + display_name + c['nc'] + c['g'],
                c['nc'] + c['b'] + class_acc_s + c['nc']
            )
        )

    if verbose > 1:
        print(''.join([' '] * 200), end='\r')
        print(
            '{:}[{:}] {:}Experiments for {:} classification finished{:}'.format(
                c['clr'] + c['c'], strftime("%m/%d/%Y - %H:%M:%S"), c['r'], c['nc'] + c['y'],
                c['nc'] + c['r'], c['b'] + name + c['nc'] + c['r'], c['nc']
            )
        )
    return acc_list, class_list


def run_attribution_experiments(
    master_seed, network_name, network, testing_files, weight_path, maps_path,
    patch_size=256, n_seeds=30, n_classes=2, saliency_batch=4, verbose=1
):
    # Choosing random runs.
    np.random.seed(master_seed)
    seeds = np.random.randint(0, 100000, n_seeds)
    # Attribution loop.
    # Due to how these things are coded, we have to run the inference again to
    # obtain the saliency maps.
    init_start = time.time()
    for test_n, seed in enumerate(seeds):
        model_path = os.path.join(
            weight_path,
            '{:}-balanced_s{:05d}_p{:03d}.pt'.format(
                network_name, seed, patch_size
            )
        )
        net = network(n_outputs=n_classes)
        net.load_model(model_path)
        net.eval()

        # While this is hard-coded (with choices for optimal waiting time),
        # the method list could be passed as a parameter.
        methods = [
            # 'Input'-based
            (InputXGradient, 'InputXGradient', 3, None),
            # Inverse
            (Deconvolution, 'Deconvolution', 3, None),
            # Perturbation-based
            # GradCAM-related
            (
                partial(LayerGradCam, layer=net.target_layer()),
                'LayerGradCam', 1, None
            ),
            (
                partial(GuidedGradCam, layer=net.target_layer()),
                'GuidedGradCam', 3, None
            ),
            (GuidedBackprop, 'GuidedBackprop', 3, None),
        ]

        # TODO: Make this generic for ANY dataset (it might not be so easy).
        # Not only do we need to run everything again, but we also need
        # to do it file by file (some of these methods require a large
        # amount of RAM).
        # While we could modify the datasets to also return coordinates and
        # hide the following code, we leave it here to show how the data is
        # loaded.
        for f, file in enumerate(testing_files):
            quadrant = file.split('_')[-1]
            saliency_set = RumexTestDataset(path, [file], patch_size)
            xmlfile = file + '.xml'
            imfile = file + '.png'
            root = et.parse(os.path.join(path, xmlfile)).getroot()
            xmlstr = et.tostring(root, encoding='utf-8', method='xml')
            xmldict = dict(xmltodict.parse(xmlstr))
            im_size = xmldict['annotation']['size']
            im_height = int(im_size['height'])
            im_width = int(im_size['width'])

            npatches_w = int(np.ceil(im_width / patch_size))
            npatches_h = int(np.ceil(im_height / patch_size))
            new_w = npatches_w * patch_size
            new_h = npatches_h * patch_size
            alpha = skio.imread(os.path.join(path, imfile))[..., 3]

            saliency_loader = DataLoader(saliency_set, saliency_batch)

            # Attribution loop.
            # We need to run each attribution method independently. The parameters
            # for these methods need to be defined on the "method list".
            for attr_m, attr_name, attr_channels, attr_args in methods:
                if attr_channels > 1:
                    heatmap = np.zeros(
                        (attr_channels, new_h, new_w), dtype=np.uint8
                    )
                else:
                    heatmap = np.zeros((new_h, new_w))
                for i, (x, y, patch_coords) in enumerate(saliency_loader):
                    time_elapsed = time.time() - init_start
                    runs_left = (len(seeds) - (test_n + 1))
                    eta = runs_left * time_elapsed / (test_n + 1)
                    if verbose > 0:
                        print(' '.join([' '] * 300), end='\r')
                        print(
                            '\033[KGenerating {:} map (batch {:d}/{:d} | '
                            'seed {:05d} [{:02d}/{:02d}]) {:} ETA {:}'.format(
                                attr_name, i + 1, len(saliency_loader),
                                seed, test_n + 1, len(seeds),
                                time_to_string(time_elapsed),
                                time_to_string(eta),
                            ), end='\r'
                        )
                    map_path = os.path.join(
                        maps_path, '{:}-balanced_s{:05d}_{:}_{:}.png'.format(
                            network_name, seed, attr_name, quadrant
                        )
                    )
                    (ini_i_lst, end_i_lst), (ini_j_lst, end_j_lst) = patch_coords
                    y = y.numpy().astype(np.uint8)
                    x_cuda = x.to(net.device)
                    x_cuda.requires_grad_()
                    pred_y = np.argmax(
                        net.inference(x.numpy()), axis=1
                    ).astype(np.uint8)
                    if attr_args is not None:
                        maps = attribution(x_cuda, attr_m(net), attr_args)
                    else:
                        maps = attribution(x_cuda, attr_m(net))
                    for ini_i, end_i, ini_j, end_j, map_k, k in zip(
                        ini_i_lst, end_i_lst, ini_j_lst, end_j_lst, maps, pred_y
                    ):
                        if k == 1:
                            if attr_channels > 1:
                                heatmap[:, ini_i:end_i, ini_j:end_j] = map_k
                            else:
                                heatmap[ini_i:end_i, ini_j:end_j] = map_k
                if attr_channels > 1:
                    final_map = np.moveaxis(
                        heatmap[:, :im_height, :im_width], 0, -1
                    )
                    final_map[alpha == 0, :] = 0
                else:
                    final_map = heatmap[:im_height, :im_width]
                    final_map[alpha == 0] = 0
                skio.imsave(map_path, final_map.astype(np.uint8))
        net = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Mean map.
    # To get a general idea on how much randomness affects results we compute
    # the mean saliency map for each method.
    for f, file in enumerate(testing_files):
        quadrant = file.split('_')[-1]
        for attr_m, attr_name, attr_channels, attr_args in methods:
            map_list = []
            for test_n, seed in enumerate(seeds):
                map_path = os.path.join(
                    maps_path, '{:}-balanced_s{:05d}_{:}_{:}.png'.format(
                        network_name, seed, attr_name, quadrant
                    )
                )
                map_list.append(skio.imread(map_path))
            map_path = os.path.join(
                maps_path, '{:}-balanced_mean_{:}_{:}.png'.format(
                    network_name, attr_name, quadrant
                )
            )
            skio.imsave(map_path, np.mean(map_list, axis=0).astype(np.uint8))
