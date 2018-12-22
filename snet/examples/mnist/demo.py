
try:
    import telemessage
    notification = True
except ImportError:
    notification = False

import torch
import os.path
import snet
from snet.examples.mnist.helper.dataloader import MNISTLoader
from localconfig import LocalConfig
from sklearn import svm
import numpy as np
import logging
import collections
import json_tricks as json
import matplotlib.pyplot as plt
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
           f"{cfg.lif_layer.winners}winners.{cfg.lif_layer.dv_th}dvth.{cfg.synapse.decay_scaling}decay-scaling." \
           f"{cfg.synapse.learn_rate_p_scaling}learn-scaling.{cfg.lif_layer.res}res." \
           f"{cfg.input.average_firing_rate}input-rate.{cfg.synapse.variation}variation." \
           f"{cfg.synapse.init_weights}-init" \
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


def train(training_images, training_labels, cfg, overwrite_check=True):
    folder = os.path.join(RESULTS_DIR, cfg_abstract(cfg))
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if overwrite_check:
            logging.warning('Result folder %s already exists, press enter to overwrite history result.' % folder)
            input()

    if notification:
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
        logging.info(network.layers['O'].spike_counts)
        network.after_batch()

        w = network.connections[('I', 'O')].weight
        dw = w - w_before
        i = torch.argmax(torch.abs(dw))
        logging.info('Max weight change = %.2e' % dw.view(-1)[i])
        logging.info('Max = %f, min = %f' % (w.max(), w.min()))

        if idx % 100 == 0:
            network.plot_weight_map(('I', 'O'), 0.01)

    # save final weight
    weight_file = os.path.join(folder, 'final_weight.pt')
    torch.save(network.connections[('I', 'O')].weight, weight_file)

    # save training config
    cfg.save(os.path.join(folder, 'training.ini'))

    if notification:
        telemessage.notify_me('Training job completed.')

    return folder


def evaluate(training_images, training_labels, testing_images, testing_labels, result_folder, trial_id=0):
    if notification:
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
    # network.plot_weight_map(('I', 'O'), 3)

    # adjust configs for testing process
    network.layers['O'].v_th *= 1
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
        # print(spike_counts)
        testing_responses.append(spike_counts)
        network.after_batch()

    # SVM training
    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(training_responses, training_labels)

    def get_accuracy(classifier, samples, sample_labels, trial_name):
        result = collections.OrderedDict()

        result['Trial'] = trial_name

        n = len(samples)

        # predict labels
        predict_labels = classifier.predict(samples)

        # predict probabilities
        probas = classifier.predict_proba(samples)

        acc_on_classes = collections.OrderedDict()
        for lbl in range(cfg.input.start_category, cfg.input.end_category + 1):
            indices = np.nonzero(sample_labels.numpy() == lbl)
            h = (predict_labels[indices] == lbl).sum()
            t = indices[0].shape[0]
            proba_mean = probas[indices].mean(axis=0)

            acc_on_classes[lbl] = (h, t, h / t, proba_mean)

        h = (predict_labels == sample_labels.numpy()).sum()
        acc_summary = (h, n, h/n)

        result['Acc. of each class'] = acc_on_classes
        result['Acc. summary'] = acc_summary

        formatted_str = ''
        # serialize result
        for key, value in result.items():
            if isinstance(value, collections.OrderedDict):
                formatted_str += f'{key}:\n'
                for k, v in value.items():
                    formatted_str += f'    {k}: {v}\n'
            else:
                formatted_str += f'{key}: {value}\n'

        return result, formatted_str

    training_result, training_summary = get_accuracy(clf, training_responses, training_labels, 'Training #%d' % trial_id)
    testing_result, testing_summary = get_accuracy(clf, testing_responses, testing_labels, 'Testing #%d' % trial_id)

    logging.info(training_summary)
    logging.info(testing_summary)

    with open(os.path.join(result_folder, 'result-%d.txt' % trial_id), 'w') as f:
        f.write(training_summary)
        f.write(testing_summary)

    if notification:
        telemessage.notify_me('Evaluation job completed, summary: \n %s \n %s' % (training_summary, testing_summary))

    return training_result, testing_result


def one_trial(config, trial_id=0, overwrite_check=True):
    tr_i, tr_l, te_i, te_l = load_mnist(config)

    result_folder = train(tr_i, tr_l, config, overwrite_check)
    # result_folder = os.path.join(RESULTS_DIR, cfg_abstract(config))

    training_result, testing_result = evaluate(tr_i, tr_l, te_i, te_l, result_folder, trial_id)

    return training_result, testing_result


def analyse_variation(trial_count):
    config = load_config()

    variation_list = np.linspace(0, 5, 11)

    results = []
    for variation in variation_list:
        config.synapse.variation = variation
        for i in range(trial_count):
            print('variation = %f, trial #%d' % (variation, i))
            training_result, testing_result = one_trial(config, i, overwrite_check=False)
            results.append((training_result, testing_result))

    with open(os.path.join(RESULTS_DIR, 'min-init-variation-results.json'), 'w') as f:
        json.dump(results, f)

    if notification:
        telemessage.notify_me("Variation trials completed.")


def visualize_variation(trial_count):
    with open(os.path.join(RESULTS_DIR, 'variation-results.json'), 'r') as f:
        results = json.load(f)
        print(results)

    variation_list = np.linspace(0, 5, 11)

    data = []

    for i, variation in enumerate(variation_list):
        data.append([result[0]['Acc. summary'][2] for result in results[trial_count*i:trial_count*(i+1)]])

    plt.hold(True)

    plt.errorbar(variation_list, np.mean(data, axis=1), np.std(data, axis=1), fmt='r')

    data = []

    for i, variation in enumerate(variation_list):
        data.append([result[1]['Acc. summary'][2] for result in results[trial_count*i:trial_count*(i+1)]])

    plt.errorbar(variation_list, np.mean(data, axis=1), np.std(data, axis=1), fmt='b')

    plt.legend(['Training', 'Testing'])
    plt.ylim((0.8, 1.0))
    plt.ylabel('Accuracy')
    plt.xlabel('Variation factor')
    plt.show()


if __name__ == '__main__':
    one_trial(load_config(), 0, False)
