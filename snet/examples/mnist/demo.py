import torch
import telemessage
import os.path
import snet
from snet.examples.mnist.helper.dataloader import MNISTLoader
from localconfig import LocalConfig
from sklearn import svm
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results/')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


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

    return f"{prefix}{cfg.input.start_category}-{cfg.input.end_category}.{cfg.network.output_neuron_number}out." \
           f"{cfg.lif_layer.winners}winners.{cfg.lif_layer.dv_th}dvth.{cfg.synapse.decay_scaling}decay-scaling" \
           f"{suffix}"


def load_mnist(cfg):
    # load MNIST dataset
    mn_loader = MNISTLoader()
    training_images, training_labels = mn_loader.load_training()
    testing_images, testing_labels = mn_loader.load_testing()

    # downsample images
    small_size = (cfg.input.image_width, cfg.input.image_height)
    training_images = mn_loader.downsample(small_size, training_images)
    testing_images = mn_loader.downsample(small_size, testing_images)

    # select desired categories
    select = ((training_labels >= cfg.input.start_category) & (training_labels <= cfg.input.end_category))\
        .nonzero().squeeze()
    training_images = torch.index_select(training_images, 0, select)
    training_labels = torch.index_select(training_labels, 0, select)

    select = ((testing_labels >= cfg.input.start_category) & (testing_labels <= cfg.input.end_category))\
        .nonzero().squeeze()
    testing_images = torch.index_select(testing_images, 0, select)
    testing_labels = torch.index_select(testing_labels, 0, select)

    return training_images, training_labels, testing_images, testing_labels


def train(training_images, training_labels, cfg):
    folder = os.path.join(RESULTS_DIR, cfg_abstract(cfg))
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        logging.warning('Result folder %s already exists, press enter to override history result.' % folder)
        input()

    # send message
    telemessage.notify_me('Training job started.')

    # build network
    network = snet.NetworkLoader().from_cfg(config=cfg)
    network.training_mode()

    for idx, image in enumerate(training_images):
        logging.info(f"Training: image #{idx}, label {training_labels[idx]}")

        w_before = network.connections[('I', 'O')].weight.clone()

        # Poisson encoding
        firing_step = network.poisson_encoding(image)
        network.layers['I'].set_step(firing_step)

        # run simulation
        network.run(cfg.input.duration_per_training_image)
        network.after_batch()

        dw = network.connections[('I', 'O')].weight - w_before
        i = torch.argmax(torch.abs(dw))
        logging.info('Max weight change = %.2e' % dw.view(-1)[i])

    # save final weight
    weight_file = os.path.join(folder, 'final_weight.pt')
    torch.save(network.connections[('I', 'O')].weight, weight_file)

    # save training config
    cfg.save(os.path.join(folder, 'training.ini'))

    telemessage.notify_me('Training job completed.')

    return folder


def evaluate(training_images, training_labels, testing_images, testing_labels, result_folder):
    # send message to my telegram account
    telemessage.notify_me('Evaluation job started.')

    # load final weight after training
    w = torch.load(os.path.join(result_folder, 'final_weight.pt'))

    # load config
    cfg = LocalConfig(os.path.join(result_folder, 'training.ini'))

    # build network
    network = snet.NetworkLoader().from_cfg(config=cfg, weight_map=w)
    network.inference_mode()

    # show weight map
    network.plot_weight_map(('I', 'O'), 3)

    # adjust configs for testing process
    network.layers['O'].winners = 4
    network.export_cfg().save(os.path.join(result_folder, 'testing.ini'))

    # inference
    # get responses on training set
    training_responses = []

    for idx, image in enumerate(training_images):
        logging.info('Inference training: image #%d, label %d' % (idx, training_labels[idx]))

        firing_step = network.poisson_encoding(image)
        network.layers['I'].set_step(firing_step)

        # Run simulation
        network.run(cfg.input.duration_per_testing_image)
        training_responses.append(network.layers['O'].spike_counts.tolist())
        network.after_batch()

    # get responses on testing set
    testing_responses = []

    for idx, image in enumerate(testing_images):
        logging.info('Inference testing: image #%d, label %d' % (idx, testing_labels[idx]))

        firing_step = network.poisson_encoding(image)
        network.layers['I'].set_step(firing_step)

        # Run simulation
        network.run(cfg.input.duration_per_testing_image)
        spike_counts = network.layers['O'].spike_counts.tolist()
        print(spike_counts)
        testing_responses.append(spike_counts)
        network.after_batch()

    # SVM training
    clf = svm.SVC(gamma='scale')
    clf.fit(training_responses, training_labels)

    for name in ['training', 'testing']:
        if name == 'training':
            responses = training_responses
            labels = training_labels
        else:
            responses = testing_responses
            labels = testing_labels

        n = len(labels)

        # SVM predicting
        predict_labels = clf.predict(responses)

        logging.info("Accuracy on %s set:" % name)

        # count
        for label in range(cfg.input.start_category, cfg.input.end_category + 1):
            indices = np.nonzero(labels.numpy() == label)
            p = predict_labels[indices]
            h = (p == label).sum()
            total = indices[0].shape[0]
            logging.info('Label %d: hit/total = %d/%d = %.4f' % (label, h, total, h / total))

        hit = (predict_labels == labels.numpy()).sum()
        logging.info('Hit/total = %d/%d = %.4f' % (hit, n, hit / n))

    telemessage.notify_me('Evaluation job completed, with test accuracy %.4f' % (hit / n))


if __name__ == '__main__':
    config = load_config()

    tr_i, tr_l, te_i, te_l = load_mnist(config)

    # result_folder = train(tr_i, tr_l, config)
    result_folder = os.path.join(RESULTS_DIR, cfg_abstract(config))

    evaluate(tr_i, tr_l, te_i, te_l, result_folder)
