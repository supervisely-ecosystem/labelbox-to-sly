import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from collections import namedtuple

import labelbox as lb
import supervisely as sly
from dotenv import load_dotenv

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ABSOLUTE_PATH)
sly.logger.debug(f"Absolute path: {ABSOLUTE_PATH}, parent dir: {PARENT_DIR}")

if sly.is_development():
    # * For convinient development, has no effect in the production.
    local_env_path = os.path.join(PARENT_DIR, "local.env")
    supervisely_env_path = os.path.expanduser("~/supervisely.env")
    sly.logger.debug(
        "Running in development mode. Will load .env files... "
        f"Local .env path: {local_env_path}, Supervisely .env path: {supervisely_env_path}"
    )

    if os.path.exists(local_env_path) and os.path.exists(supervisely_env_path):
        sly.logger.debug("Both .env files exists. Will load them.")
        load_dotenv(local_env_path)
        load_dotenv(supervisely_env_path)
    else:
        sly.logger.warning("One of the .env files is missing. It may cause errors.")

api: sly.Api = sly.Api.from_env()

TEMP_DIR = os.path.join(PARENT_DIR, "temp")

# * Directory, where downloaded and convertted to coco Labelbox data will be stored.
COCO_DIR = os.path.join(TEMP_DIR, "coco")

# # * Directory, where converted Supervisely data will be stored.
SLY_DIR = os.path.join(TEMP_DIR, "sly")

sly.fs.mkdir(COCO_DIR, remove_content_if_exists=True)
sly.fs.mkdir(SLY_DIR, remove_content_if_exists=True)
sly.logger.debug(
    f"TEMP_DIR: {TEMP_DIR}, COCO_DIR: {COCO_DIR}, SLY_DIR: {SLY_DIR}"
)

DEFAULT_API_ADDRESS = "https://app.labelbox.com/"

EXPORT_SOURCE = "project"  # "project", "dataset" or "model_run"
ENTITY_TYPE = "image"  # "video" or "image"

GEOMETRIES_MAPPING = {
    "BBOX": sly.Rectangle,
    "LINE": sly.Polyline,
    "RASTER_SEGMENTATION": sly.Bitmap,
    "SEGMENTATION": sly.Bitmap,
    "POLYGON": sly.Polygon,
    "POINT": sly.Point,
}


class State:
    def __init__(self):
        self.selected_team = sly.env.team_id()
        self.selected_workspace = sly.env.workspace_id()

        # Will be set to True, if the app will be launched from .env file in Supervisely.
        self.loaded_from_env = False

        # Labelbox credentials to access the API.
        self.labelbox_api_address = None

        self.labelbox_api_key = None
        self.client = None

        self.projects = {}
        self.selected_projects = []
        self.selected_projects_ids = []
        self.proccessed_projects = []

        # Will be set to False if the cancel button will be pressed.
        # Sets to True on every click on the "Copy" button.
        self.continue_copying = True
        self.export_from = None

        self.export_params = {
            "attachments": True,
            "metadata_fields": True,
            "data_row_details": True,
            "project_details": True,
            "label_details": True,
            "performance_details": True,
        }
        self.export_tasks = {}

    def clear_labelbox_credentials(self):
        """Clears the Labelbox credentials and sets them to None."""

        sly.logger.debug("Clearing Labelbox credentials...")
        self.labelbox_api_address = None
        self.labelbox_api_key = None

    def load_from_env(self):
        """Downloads the .env file from Supervisely and reads the Labelbox credentials from it."""
        if not LABELBOX_ENV_TEAMFILES:
            sly.logger.warning("No .env file provided. It should be provided in the next step.")
            return
        try:
            api.file.download(STATE.selected_team, LABELBOX_ENV_TEAMFILES, LABELBOX_ENV_FILE)
        except Exception as e:
            sly.logger.warning(f"Failed to download .env file: {e}")
            return

        sly.logger.debug(".env file downloaded successfully. Will read the credentials.")

        load_dotenv(LABELBOX_ENV_FILE)

        self.labelbox_api_address = os.getenv("LABELBOX_API_ADDRESS", DEFAULT_API_ADDRESS)
        self.labelbox_api_key = os.getenv("LABELBOX_API_KEY") or os.getenv("LB_API_KEY")
        sly.logger.debug(
            "Labelbox credentials readed successfully. "
            f"API address: {self.labelbox_api_address}, API key is hidden in logs. "
            "Will check the connection."
        )
        self.loaded_from_env = True

    def connect_to_labelbox(self):
        """Connects to the Labelbox API."""

        try:
            self.client = lb.Client(api_key=self.labelbox_api_key)
            sly.logger.debug("Connected to the Labelbox API.")
        except Exception as e:
            sly.logger.error(f"Exception when calling Roboflow API: {e}")
            return None
        return self.client

    def set_source_type(self, source_type: Literal["project", "dataset", "model_run"]):
        """Sets the source type of the export."""
        if source_type not in ["project", "dataset", "model_run"]:
            raise ValueError(f"Unknown source type: {source_type}")
        self.export_from = source_type
        sly.logger.debug(f"Selected source: {source_type}")


STATE = State()
sly.logger.debug(
    f"Selected team: {STATE.selected_team}, selected workspace: {STATE.selected_workspace}"
)

# * Local path to the .env file with credentials, after downloading it from Supervisely.
LABELBOX_ENV_FILE = os.path.join(PARENT_DIR, "labelbox.env")
sly.logger.debug(f"Path to the local labelbox.env file: {LABELBOX_ENV_FILE}")

# * Path to the .env file with credentials (on Team Files).
# While local development can be set in local.env file with: context.slyFile = "/.env/labelbox.env"
LABELBOX_ENV_TEAMFILES = sly.env.file(raise_not_found=False)
sly.logger.debug(f"Path to the TeamFiles from environment: {LABELBOX_ENV_TEAMFILES}")

CopyingStatus = namedtuple("CopyingStatus", ["copied", "error", "waiting", "working", "ready"])
COPYING_STATUS = CopyingStatus("✅ Copied", "❌ Error", "⏳ Waiting", "🔄 Working", "ℹ️ Ready to start")

if LABELBOX_ENV_FILE:
    sly.logger.debug(".env file is provided, will try to download it.")
    STATE.load_from_env()
