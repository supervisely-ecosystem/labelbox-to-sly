import os
from datetime import datetime

import labelbox as lb
import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, Flexbox, Progress, Table, Text

import src.globals as g
from src.converters import coco_to_supervisely, process_video_project
from src.labelbox_api import download_coco_format_project

COLUMNS = [
    "COPYING STATUS",
    "UID",
    "NAME",
    "CREATED",
    "UPDATED",
    "LABELBOX URL",
    "SUPERVISELY URL",
]

projects_table = Table(fixed_cols=3, per_page=20, sort_column_id=1)
projects_table.hide()

copy_button = Button("Copy", icon="zmdi zmdi-copy")
stop_button = Button("Stop", icon="zmdi zmdi-stop", button_type="danger")
stop_button.hide()

buttons_flexbox = Flexbox([copy_button, stop_button])

copying_progress = Progress()
processing_notification = Text(text="Processing. It may take a while...", status="info")
good_results = Text(status="success")
bad_results = Text(status="error")
processing_notification.hide()
good_results.hide()
bad_results.hide()

card = Card(
    title="3️⃣ Copying",
    description="Copy selected projects from Labelbox to Supervisely.",
    content=Container(
        [
            projects_table,
            buttons_flexbox,
            copying_progress,
            processing_notification,
            good_results,
            bad_results,
        ]
    ),
    collapsable=True,
)
card.lock()
card.collapse()


def build_projects_table() -> None:
    """Fills the table with projects from Labelbox API.
    Uses global g.STATE.selected_projects to get the list of projects to show.
    """
    sly.logger.debug("Building projects table...")
    projects_table.loading = True
    rows = []

    for project in g.STATE.selected_projects:
        project_url = g.DEFAULT_API_ADDRESS + "projects" + f"/{project.uid}" + "/overview"

        rows.append(
            [
                g.COPYING_STATUS.waiting,
                project.uid,
                project.name,
                datetime_to_str(project.created_at),
                datetime_to_str(project.updated_at),
                f'<a href="{project_url}" target="_blank">{project_url}</a>',
                "",
            ]
        )

    sly.logger.debug(f"Prepared {len(rows)} rows for the projects table.")

    projects_table.read_json(
        {
            "columns": COLUMNS,
            "data": rows,
        }
    )

    projects_table.loading = False
    projects_table.show()

    sly.logger.debug("Projects table is built.")


def datetime_to_str(datetime_object: datetime) -> str:
    """Converts datetime object to string for HTML table.

    :param datetime_object: datetime object
    :type datetime_object: datetime
    :return: HTML-formatted string
    :rtype: str
    """
    return datetime_object.strftime("<b>%Y-%m-%d</b> %H:%M:%S")


@copy_button.click
def start_copying() -> None:
    """Main function for copying projects from Labelbox to Supervisely.

    1. Tries to download the project from Labelbox API and save it as JSON file.
    2. Read JSOn file data, converts it to Supervisely format and uploads it to Supervisely.
    3. Updates cells in the projects table by project ID.
    4. Clears the download and upload directories.
    5. Stops the application.
    """
    sly.logger.debug(f"Copying button is clicked. Selected projects: {g.STATE.selected_projects}")

    stop_button.show()
    processing_notification.show()
    copy_button.text = "Copying..."
    g.STATE.continue_copying = True

    succesfully_uploaded = 0
    uploaded_with_errors = 0

    # progress_bar = sly.Progress("Copying...", len(g.STATE.selected_projects))
    with copying_progress(total=len(g.STATE.selected_projects), message="Copying...") as copy_pbar:
        for project in g.STATE.selected_projects:
            if not g.STATE.continue_copying:
                sly.logger.info("Stop button pressed. Will stop copying.")
                break
            if project.uid in g.STATE.proccessed_projects:
                sly.logger.debug(f"Project '{project.name}' was already processed.")
                continue
            sly.logger.debug(f"Copying project '{project.name}'")
            update_cells(project.uid, new_status=g.COPYING_STATUS.working)

            upload_status = False
            if project.media_type == lb.MediaType.Video:
                sly.logger.debug(f"Project '{project.name}' has video labels.")
                try:
                    sly_project = process_video_project(project)
                except Exception as e:
                    sly.logger.error(f"Can't process the project '{project.name}'. {e}")
                    sly_project = False
                if sly_project:
                    set_project_url(project, sly_project.id)
                    upload_status = True
                else:
                    upload_status = False
            elif project.media_type == lb.MediaType.Image:
                try:
                    project_src_dir = download_coco_format_project(project)
                except Exception as e:
                    sly.logger.error(f"Can't process the project {project.name}: {e}")
                    project_src_dir = False
                if not project_src_dir:
                    sly.logger.warning(f"Project {project.name} was not downloaded.")
                    update_cells(project.uid, new_status=g.COPYING_STATUS.error)
                    uploaded_with_errors += 1
                    copy_pbar.update(1)
                    continue
                project_dst_dir = os.path.join(g.SLY_DIR, project.name)
                sly.fs.mkdir(project_dst_dir, remove_content_if_exists=True)

                upload_status = convert_and_upload(project_src_dir, project_dst_dir, project)

            if upload_status:
                sly.logger.info(f"Project {project.name} was uploaded successfully.")
                new_status = g.COPYING_STATUS.copied
                succesfully_uploaded += 1
                g.STATE.proccessed_projects.append(project.uid)
            else:
                sly.logger.warning(f"Project {project.name} was not uploaded.")
                new_status = g.COPYING_STATUS.error
                uploaded_with_errors += 1

            update_cells(project.uid, new_status=new_status)
            sly.logger.debug(f"Updated project {project.name} in the projects table.")

            sly.logger.info(f"Finished processing project {project.name}.")

            copy_pbar.update(1)

    if succesfully_uploaded:
        good_results.text = f"Succesfully uploaded {succesfully_uploaded} projects."
        good_results.show()
    if uploaded_with_errors:
        bad_results.text = f"Erorrs occured while processing {uploaded_with_errors} projects."
        bad_results.show()

    copy_button.text = "Copy"
    stop_button.hide()
    processing_notification.hide()

    sly.logger.info(f"Finished copying {len(g.STATE.selected_projects)} projects.")

    if sly.is_development():
        # * For debug purposes it's better to save the data from Labelbox API.
        sly.logger.debug(
            "Development mode, will not stop the application. "
            "And NOT clean download and upload directories."
        )
        return

    sly.fs.clean_dir(g.COCO_DIR)
    sly.fs.clean_dir(g.SLY_DIR)

    sly.logger.info(
        f"Removed content from '{g.SLY_DIR}' and '{g.COCO_DIR}'." "Will stop the application."
    )

    from src.main import app

    app.stop()


def convert_and_upload(src_dir: str, dst_dir: str, project: lb.Project) -> bool:
    """Converts project from COCO format to Supervisely format and uploads it to Supervisely.

    :param src_dir: path to the directory with COCO format project
    :type src_dir: str
    :param dst_dir: path to the directory where the converted project will be saved
    :type dst_dir: str
    :param project: project from Labelbox API
    :type project: lb.Project
    :return: True if project was uploaded successfully, False otherwise
    :rtype: bool
    """
    try:
        coco_to_supervisely(src_dir, dst_dir, ignore_bbox=True)
    except Exception as e:
        sly.logger.warning(f"Can't convert project {project.name} to Supervisely: {e}")
        return False

    try:
        (sly_id, sly_name) = sly.Project.upload(
            dst_dir, g.api, g.STATE.selected_workspace, project.name, log_progress=True
        )
    except Exception as e:
        sly.logger.warning(f"Can't upload project {project.name} to Supervisely: {e}")
        return False

    set_project_url(project, sly_id)

    sly.logger.debug(f"Project {sly_name} was processed successfully.")
    return True


def update_cells(project_id: int, **kwargs) -> None:
    """Updates cells in the projects table by project ID.
    Possible kwargs:
        - new_status: new status for the project
        - new_url: new Supervisely URL for the project

    :param project_id: project ID in Labelbox for projects table to update
    :type project_id: int
    """
    key_cell_value = project_id
    key_column_name = "UID"
    if kwargs.get("new_status"):
        column_name = "COPYING STATUS"
        new_value = kwargs["new_status"]
    elif kwargs.get("new_url"):
        column_name = "SUPERVISELY URL"
        url = kwargs["new_url"]
        new_value = f"<a href='{url}' target='_blank'>{url}</a>"

    projects_table.update_cell_value(key_column_name, key_cell_value, column_name, new_value)


def set_project_url(lb_project: int, sly_project_id: int) -> None:
    """Sets the project URL in the projects table by project ID.

    :param project_id: project ID in Labelbox for projects table to update
    :type project_id: int
    :param sly_id: project ID in Supervisely
    :type sly_id: int
    """
    project_info = g.api.project.get_info_by_id(sly_project_id)
    try:
        new_url = sly.utils.abs_url(project_info.url)
    except Exception:
        new_url = project_info.url
    sly.logger.debug(f"New URL for images project: {new_url}")
    update_cells(lb_project.uid, new_url=new_url)


@stop_button.click
def stop_copying() -> None:
    """Stops copying process by setting continue_copying flag to False."""
    sly.logger.debug("Stop button is clicked.")

    g.STATE.continue_copying = False
    copy_button.text = "Stopping..."

    stop_button.hide()
