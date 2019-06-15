import argparse
import sys
import os
from six.moves import urllib
import tarfile


def download(data_url, data_dir):

  if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

  download_tar_name = os.path.join("/tmp", os.path.basename(data_url))

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading to %s %.1f%%' % (
        download_tar_name, 100.0 * count * block_size / total_size))
    sys.stdout.flush()

  local_tar_name, _ = urllib.request.urlretrieve(data_url,
                                                 download_tar_name,
                                                 _progress)

  print("\nExtracting dataset to " + data_dir)
  tarfile.open(local_tar_name, 'r:gz').extractall(data_dir)


def check_data(config):
  if config.mode != "infer" and config.mode != "export":
    if config.mode == "tune":
      for file in config.train_dataset_meta:    
        if not os.path.isfile(file):
          assert False, file + " is not available. Please run demo/download_data.py to download data."
      for file in config.eval_dataset_meta:    
        if not os.path.isfile(file):
          assert False, file + " is not available. Please run demo/download_data.py to download data."          
    else:
      for file in config.dataset_meta:
        if not os.path.isfile(file):
          assert False, "Data is not available. Please run demo/download_data.py to download data"
    print("Passed data check.")   



def main():
  sys.path.append('.')

  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--data_url",
                      help="URL for downloading data",
                      default="")
  parser.add_argument("--data_dir",
                      help="Path for data folder",
                      default="")

  args = parser.parse_args()

  args.data_dir = os.path.expanduser(args.data_dir)
  download(args.data_url, args.data_dir)

if __name__ == "__main__":
  main()