import sklearn.utils
import json
import os
from urllib.request import urlretrieve
import tarfile
import shutil
import glob

current_path  = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(current_path, 'datasets')
PREPEARD_DATASET = 'data'
TAR_DIR = os.path.join(DATASETS_DIR, 'tar')
EMAILS_JSON = 'emails.json'
LABELS_JSON = 'labels.json'

SPAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'
EASY_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'
HARD_HAM_URL = 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'

train_test_spilt = 0.8


def download_dataset(url):
    """download and unzip data from a url into the specified path"""

    # create directory if it doesn't exist
    if not os.path.isdir(TAR_DIR):
        os.makedirs(TAR_DIR)

    filename = url.rsplit('/', 1)[-1]
    tarpath = os.path.join(TAR_DIR, filename)

    # download the tar file if it doesn't exist
    try:
        tarfile.open(tarpath)
    except:
        urlretrieve(url, tarpath)

    with tarfile.open(tarpath) as tar:
        dirname = os.path.join(DATASETS_DIR, tar.getnames()[0])
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)
        tar.extractall(path=DATASETS_DIR)

        cmds_path = os.path.join(dirname, 'cmds')
        if os.path.isfile(cmds_path):
            os.remove(cmds_path)

    return dirname


def load_dataset(dirpath):
    """load emails from the specified directory"""

    files = []
    filepaths = glob.glob(dirpath + '/*')
    for path in filepaths:
        with open(path, 'rb') as f:
            byte_content = f.read()
            str_content = byte_content.decode('utf-8', errors='ignore')
            files.append(str_content)
    return files


def load():
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

    preprocesd_folders_exists = os.path.exists(os.path.join(DATASETS_DIR, PREPEARD_DATASET)) and os.path.join(
        DATASETS_DIR, PREPEARD_DATASET, EMAILS_JSON)

    if preprocesd_folders_exists:
        with open(os.path.join(DATASETS_DIR, PREPEARD_DATASET, EMAILS_JSON)) as fp:
            X = json.load(fp)

    else:
        if not os.path.exists(os.path.join(DATASETS_DIR, PREPEARD_DATASET)):
            os.mkdir(os.path.join(DATASETS_DIR, PREPEARD_DATASET))

        # download
        spam_dir = download_dataset(SPAM_URL)
        easy_ham_dir = download_dataset(EASY_HAM_URL)
        hard_ham_dir = download_dataset(HARD_HAM_URL)

        # load the datasets
        spam = load_dataset(spam_dir)
        easy_ham = load_dataset(easy_ham_dir)
        hard_ham = load_dataset(hard_ham_dir)

        # just removing something unnececary
        remove_header = lambda x: x[x.index('\n\n'):]
        spam = list(map(remove_header, spam))
        easy_ham = list(map(remove_header, easy_ham))
        hard_ham = list(map(remove_header, hard_ham))

        # shuffle the dataset
        spam = sklearn.utils.shuffle(spam, random_state=42)
        easy_ham = sklearn.utils.shuffle(easy_ham, random_state=42)
        hard_ham = sklearn.utils.shuffle(hard_ham, random_state=42)

        # create the full dataset
        X = {'spam': spam, 'ham': easy_ham + hard_ham}

        # save data
        with open(os.path.join(DATASETS_DIR, PREPEARD_DATASET, EMAILS_JSON), 'w') as fp:
            json.dump(X, fp)

    data_train = {
        'spam': X['spam'][:int(train_test_spilt * len(X['spam']))],
        'ham': X['ham'][:int(train_test_spilt * len(X['ham']))],
    }

    data_test = {
        'spam': X['spam'][int(train_test_spilt * len(X['spam'])):],
        'ham': X['ham'][int(train_test_spilt * len(X['ham'])):],
    }
    return data_train, data_test


if __name__ == '__main__':
    X_train, X_test = load()
    pass
