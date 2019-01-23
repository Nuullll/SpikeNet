
import os
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from localconfig import LocalConfig
import snet
import torch
import logging
logging.basicConfig(level=logging.WARNING)

try:
    import telemessage
    notification = True
except ImportError:
    notification = False


DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def load_bimnist(train=True):

    dataset = MNIST(DATASET_DIR, train=train, download=True, transform=transforms.Compose([
        # transforms.Resize((11, 11)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: (x > 0.5).squeeze(0))
    ]))

    return dataset


def load_config():
    return LocalConfig(os.path.join(os.path.dirname(__file__), 'network.ini'))


def cfg_abstract(cfg, prefix='', suffix=''):
    """
    :return:    Config abstract string.
    """
    if not prefix == '':
        prefix = prefix + '.'
    if not suffix == '':
        suffix = '.' + suffix

    return f"{prefix}{cfg.input.start_category}-{cfg.input.end_category}." \
           f"{cfg.network.output_neuron_number}out." \
           f"{cfg.input.pattern_firing_rate}pattern." \
           f"{cfg.synapse.tau_p}stdp_tau" \
           f"{suffix}"


def train(training_dataset, cfg, overwrite_check=True):
    folder = os.path.join(RESULTS_DIR, cfg_abstract(cfg))
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if overwrite_check:
            logging.warning('Result folder %s already exists, press enter to overwrite history result.' % folder)
            input()
    #
    # if notification:
    #     # send message
    #     telemessage.notify_me('Training job started.')

    # build network
    network = snet.NetworkLoader().from_cfg(config=cfg)
    network.training_mode()

    labels = list(range(cfg.input.start_category, cfg.input.end_category + 1))

    label_history = []

    idx = 0
    for image, label in training_dataset:
        if label not in labels:
            continue

        print('[%d] Label=%d' % (idx, label))

        # input image
        network.input_image(image)

        # run simulation
        if network.run(cfg.input.duration_per_training_image):
            # noise stimuli
            noise = 1 - image.view(-1)
            network.input_image(noise)
            network.layers['I'].image_norm /= 20
            network.run(20)
        label_history.append(label.item())

        # plt.plot(network.monitors['O'].record['v'].numpy())
        # plt.show()
        counts = network.layers['O'].spike_counts.clone()

        # Extra run
        # while counts.sum() == 0:
        #     network.after_batch(keep_count=True)
        #     network.run(cfg.input.duration_per_training_image)
        #     label_history.append(label.item())
        #
        #     print('@Extra')
        #     counts = network.layers['O'].spike_counts.clone()
        #
        #     break

        print(network.time)
        print(counts)
        print(network.layers['O'].v_th)

        network.after_batch()

        if idx % 1000 == 0:
            plt.figure(4)
            plt.clf()
            spike_events = network.layers['O'].activity_history.nonzero()
            plt.scatter(spike_events.numpy()[:, 0], spike_events.numpy()[:, 1],
                        c=torch.tensor(label_history)[spike_events[:, 0]].numpy())
            plt.pause(0.01)

        idx += 1

    # # save final weight
    # weight_file = os.path.join(folder, 'final_weight.pt')
    torch.save(network.connections[('I', 'O')].weight, os.path.join(folder, 'final_weight.pt'))
    torch.save(network.layers['O'].v_th, os.path.join(folder, 'v_th.pt'))
    #
    # # save training config
    cfg.save(os.path.join(folder, 'training.ini'))
    #
    # if notification:
    #     telemessage.notify_me('Training job completed.')
    #
    return folder


def test(training_dataset, testing_dataset, folder):
    # load config
    cfg = LocalConfig(os.path.join(folder, 'training.ini'))
    weight = torch.load(os.path.join(folder, 'final_weight.pt'))
    v_th = torch.load(os.path.join(folder, 'v_th.pt'))

    network = snet.NetworkLoader().from_cfg(config=cfg, weight_map=weight)
    network.inference_mode()

    network.layers['O'].v_th = v_th

    plt.figure(1)
    network.plot_weight_map(('I', 'O'), 0.1)

    labels = list(range(cfg.input.start_category, cfg.input.end_category + 1))

    response_map = torch.zeros(len(labels), len(v_th))

    label_history = []

    idx = 0
    for image, label in training_dataset:
        if label not in labels:
            continue

        print('[%d] Label=%d' % (idx, label))

        # input image
        network.input_image(image)

        # run simulation
        network.run(cfg.input.duration_per_testing_image)
        label_history.append(label.item())

        # plt.plot(network.monitors['O'].record['v'].numpy())
        # plt.show()
        counts = network.layers['O'].spike_counts.clone()

        if counts.sum() > 0:
            response_map[label] += counts.float() / counts.sum()

        print(counts)

        for post in range(len(v_th)):
            print('Neuron #%d' % post)

            score = response_map[:, post].clone()

            _, max_ind = score.max(dim=0)

            print('Labeled as %d' % labels[max_ind])
            print('Proba vec:', score)

        network.after_batch()

        idx += 1

    torch.save(response_map, os.path.join(folder, 'response.pt'))
    # response_map = torch.load(os.path.join(folder, 'response.pt'))

    neuron_activity = response_map.sum(0)

    score_map = response_map / neuron_activity * neuron_activity.min() / neuron_activity

    wrong_count = 0
    # test accuracy
    idx = 0

    # for post in range(len(v_th)):
    #     print('Neuron #%d' % post)
    #
    #     score = score_map[:, post].clone()
    #
    #     max_val, max_ind = score.max(dim=0)
    #
    #     if max_val < 0.8:
    #         print('Pruned.')
    #         network.connections[('I', 'O')].weight[:, post] = torch.zeros(28, 28).view(-1)

    for image, label in testing_dataset:
        if label not in labels:
            continue
        print("[Test #%d]" % idx)

        network.input_image(image)
        network.run(config.input.duration_per_testing_image)

        counts = network.layers['O'].spike_counts.clone()

        network.after_batch()

        if counts.sum() > 0:
            score = torch.matmul(score_map, counts.squeeze(0).float() / counts.sum())

            _, max_ind = score.max(dim=0)

            # _, max_ind = counts.max(0)
            # max_ind = neuron_labels[max_ind]

            predict = labels[max_ind]

            if not predict == label:
                wrong_count += 1
                print("Expected: %d, predicted: %d" % (label, predict))
                print("Count: ", counts)
                print("Score: ", score)
                print("Wrong: %d/%d" % (wrong_count, idx+1))

        idx += 1


def play():
    cfg = load_config()
    dataset = load_bimnist()
    network = snet.NetworkLoader().from_cfg(config=cfg)
    network.inference_mode()

    legends = []
    ind = []
    for idx in range(8):
        i = random.choice(range(10000))
        ind.append(i)
        image, label = dataset[i]
        legends.append(str(label.item()))

        synapse = network.connections[('I', 'O')]
        synapse.weight[:, idx] = torch.rand_like(image.view(-1)) * synapse.w_max * image.view(-1)

    plt.figure(1)
    plt.clf()
    network.plot_weight_map(("I", "O"), 0.03)
    for i in ind:
        image, label = dataset[i]

        output = network.layers['O']
        output.v = torch.ones_like(output.v) * output.v_rest
        network.input_image(image)

        network.run(1000)

    plt.figure(2)
    plt.plot(network.monitors['O'].record['v'].numpy())
    plt.legend(legends)
    plt.show()


if __name__ == '__main__':
    config = load_config()
    ds = load_bimnist()
    result_folder = train(ds, config)
    # test(ds, load_bimnist(False), os.path.join(RESULTS_DIR, '0-2.12out.1.0pattern.5.0stdp_tau'))
    test(ds, load_bimnist(False), result_folder)
    # play()
