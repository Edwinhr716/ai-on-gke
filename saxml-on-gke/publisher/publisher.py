"""Publisher logic for SaxML deployment on Kubernetes.
Reads model data from a ConfigMap that has been mounted to the container.

On startup, it verfies whether or not the model has already been loaded. 
Publishes the model if not.
If there is a different model already loaded, it unpublishes the old one and 
publishes the new one.

If the data in the ConfigMap changes, it unpublishes the current model that 
has been loaded, and publishes a model with the new information.
"""

import configparser
import itertools
import os
import time

import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent
from watchdog.events import FileSystemEventHandler


TWELVE_MINUTES = 720

class EventHandler(FileSystemEventHandler):
  """Class to detect changes in files."""

  def __init__(self, original_model: dict[str, str]) -> None:
    self.original_model = original_model

  def on_deleted(self, event: FileSystemEvent):
    """When ConfigMap is updated, publish.conf is deleted and created again."""
    print(
        "Model information in config map has changed, unpublishing"
        f" {self.original_model['model']}"
    )
    unpublish(self.original_model["model"])
    new_model = get_config_map()
    print(f"Publishing model: {new_model['model']}")
    publish(new_model)
    self.original_model = new_model


def are_same_model(
    installed_model: dict[str, str],
    new_model: dict[str, str]
) -> bool:
  if installed_model["model"] != new_model["model"]:
    print(
        f"checkpoint does not match {installed_model['model']},"
        f" {new_model['model']}"
    )
    return False
  if installed_model["model_path"] != new_model["model_path"]:
    print(
        f"checkpoint does not match {installed_model['model_path']},"
        f" {new_model['model_path']}"
    )
    return False
  if installed_model["checkpoint"] != new_model["checkpoint"]:
    print(
        f"checkpoint does not match {installed_model['checkpoint']},"
        f" {new_model['checkpoint']}"
    )
    return False
  return True


def get_config_map() -> dict[str, str]:
  """Returns model information that is stored in the config map."""
  config = configparser.ConfigParser()
  config_map_path = "/publish/config/publish.conf"
  with open(config_map_path) as fp:
    config.read_file(itertools.chain(["[global]"], fp), source=config_map_path)
  model_information = dict(config.items("global"))
  if "model" not in model_information:
    model_information["model"] = os.environ["SAX_CELL"] + "/" + model_information["model_path"].split(".")[-1].lower()
  return model_information


def get_installed_models() -> list[str]:
  list_all_url = "http://localhost:8888/listall"
  sax_cell = os.environ["SAX_CELL"]
  list_all_response = requests.get(list_all_url, json={"sax_cell": sax_cell})
  return list_all_response.json()


def get_model_information(model: str) -> dict[str, str]:
  list_cell_url = "http://localhost:8888/listcell"
  list_sax_cell_response = requests.get(list_cell_url, json={"model": model})
  return list_sax_cell_response.json()


def publish(model_information: dict[str, str]):
  publish_url = "http://localhost:8888/publish"
  publish_response = requests.post(
      publish_url,
      json={
          "model": model_information["model"],
          "model_path": model_information["model_path"],
          "checkpoint": model_information["checkpoint"],
          "replicas": 1,
      },
  )
  if publish_response.status_code != 200:
    print(f"Error publising model: {publish_response.status_code}")


def unpublish(model: str):
  unpublish_url = "http://localhost:8888/unpublish"
  unpublish_response = requests.post(unpublish_url, json={"model": model})
  if unpublish_response.status_code != 200:
    print(f"Error unpublising model: {unpublish_response.status_code}")


def check_published_models() -> dict[str, str]:
  time.sleep(15)
  new_model = get_config_map()
  installed_models = get_installed_models()
  if not installed_models:
    print(f"publishing {new_model['model']}")
    publish(new_model)
    return new_model
  # LWS and singlehost only support one model
  installed_model = installed_models[0]
  if installed_model != new_model["model"]:
    print(
        f"Model already loaded, unpublishing {installed_model} and publishing"
        f" {new_model['model']}"
    )
    # Can't make unpublish call until the model server has loaded the model
    # stored in the bucket. That takes around twelve minutes.
    # There is an edge case where the user changes the configmap during the 12 minutes,
    # which won't be detected 
    time.sleep(TWELVE_MINUTES)
    unpublish(installed_model)
    time.sleep(15)
    publish(new_model)
    return new_model

  model_information = get_model_information(installed_model)
  if are_same_model(model_information, new_model):
    print("model has already been installed")
    return new_model

  unpublish(model_information)
  time.sleep(15)
  publish(new_model)
  return new_model


def publish_on_update(installed_model: dict[str, str]):
  event_handler = EventHandler(installed_model)
  observer = Observer()
  observer.schedule(event_handler, "/publish/config")
  observer.start()
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    observer.stop()
  observer.join()


published_model = check_published_models()
publish_on_update(published_model)
