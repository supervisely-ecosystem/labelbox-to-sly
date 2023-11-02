import supervisely as sly
from supervisely.app.widgets import Button, Card, Container, Field, Input, Text

import src.globals as g
import src.ui.selection as selection
import src.ui.copying as copying

labelbox_api_address_input = Input(
    minlength=10,
    value="https://app.labelbox.com",
    placeholder="for example: https://app.labelbox.com",
    readonly=False,
)
labelbox_api_address_field = Field(
    title="Labelbox API address",
    description="Address of the Labelbox API to connect.",
    content=labelbox_api_address_input,
)

labelbox_api_key_input = Input(minlength=1, type="password", placeholder="for example: admin")
labelbox_api_key_field = Field(
    title="Labelbox API key",
    description="Labelbox private API key to connect.",
    content=labelbox_api_key_input,
)

connect_button = Button("Connect to Labelbox")
connect_button.disable()
change_connection_button = Button("Change settings")
change_connection_button.hide()

load_from_env_text = Text("Connection settings was loaded from .env file.", status="info")
load_from_env_text.hide()
connection_status_text = Text()
connection_status_text.hide()


card = Card(
    title="1️⃣ Labelbox connection",
    description="Enter your Labelbox connection settings and check the connection.",
    content=Container(
        [
            labelbox_api_address_field,
            labelbox_api_key_field,
            connect_button,
            load_from_env_text,
            connection_status_text,
        ]
    ),
    content_top_right=change_connection_button,
    collapsable=True,
)


def connected() -> None:
    """Changes the state of the widgets if the app successfully connected to the Labelbox API.
    Launches the process of filling the transfer with projects from Labelbox API."""

    sly.logger.debug("Status changed to connected, will change widget states.")
    labelbox_api_address_input.disable()
    labelbox_api_key_input.disable()
    connect_button.disable()

    card.collapse()
    selection.card.unlock()
    selection.card.uncollapse()

    change_connection_button.show()
    connection_status_text.status = "success"
    connection_status_text.text = f"Successfully connected to {formatted_connection_settings()}."

    connection_status_text.show()
    selection.fill_transfer_with_projects()


def disconnected(with_error=False) -> None:
    """Changes the state of the widgets if the app disconnected from the Labelbox API.
    Depending on the value of the with_error parameter, the status text will be different.

    :param with_error: if the app was disconnected from server or by a pressing change button, defaults to False
    :type with_error: bool, optional
    """

    sly.logger.debug(
        f"Status changed to disconnected with error: {with_error}, will change widget states."
    )

    labelbox_api_address_input.enable()
    labelbox_api_key_field.enable()
    connect_button.enable()

    card.uncollapse()
    selection.card.lock()
    selection.card.collapse()
    copying.card.lock()
    copying.card.collapse()

    change_connection_button.hide()

    if with_error:
        connection_status_text.status = "error"
        connection_status_text.text = (
            f"Failed to connect to {formatted_connection_settings()}. Please check the credentials."
        )

    else:
        connection_status_text.status = "warning"
        connection_status_text.text = f"Disconnected from {formatted_connection_settings()}."

    g.STATE.clear_labelbox_credentials()
    connection_status_text.show()


def formatted_connection_settings() -> str:
    """Returns HTML-formatted string with the Labelbox connection settings (server address, username).

    :return: HTML-formatted string
    :rtype: str
    """
    return f'<a href="{g.STATE.labelbox_api_address}">{g.STATE.labelbox_api_address}</a>'


def change_connect_button_state(_: str) -> None:
    """Enables the connect button if all the required fields are filled,
    otherwise disables it.

    :param _: Unused (value from the widget)
    :type input_value: str
    """
    if all(
        [
            labelbox_api_address_input.get_value(),
            labelbox_api_key_input.get_value(),
        ]
    ):
        connect_button.enable()

    else:
        connect_button.disable()


labelbox_api_address_input.value_changed(change_connect_button_state)
labelbox_api_key_input.value_changed(change_connect_button_state)


@change_connection_button.click
def change_connection_settings() -> None:
    """Changes the state of the widgets if the user wants to change the connection settings."""

    sly.logger.debug("Changing connection settings...")

    disconnected(with_error=False)

    labelbox_api_address_input.enable()
    labelbox_api_key_input.enable()
    connect_button.enable()

    change_connection_button.hide()
    connection_status_text.hide()


@connect_button.click
def try_to_connect() -> None:
    """Save the Labelbox credentials from the widgets to the global State and try to connect to the Labelbox API.
    Depending on the result of the connection, the state of the widgets will change."""

    g.STATE.labelbox_api_address = labelbox_api_address_input.get_value()
    g.STATE.labelbox_api_key = labelbox_api_key_input.get_value()

    sly.logger.debug(
        f"Saved Labelbox credentials in global State. "
        f"API address: {g.STATE.labelbox_api_address}, "
    )

    is_connected = g.STATE.connect_to_labelbox()

    if is_connected is not None:
        connected()
    else:
        disconnected(with_error=True)


if g.STATE.loaded_from_env:
    sly.logger.debug('The application was started with the "Load from .env" option.')

    load_from_env_text.show()

    labelbox_api_address_input.set_value(g.STATE.labelbox_api_address)
    labelbox_api_key_input.set_value(g.STATE.labelbox_api_key)
    connect_button.enable()

    is_connected = g.STATE.connect_to_labelbox()

    if is_connected:
        sly.logger.info(
            f"Connection to Labelbox server {g.STATE.labelbox_api_address} was successful."
        )

        connected()

    else:
        sly.logger.warning(f"Connection to Labelbox server {g.STATE.labelbox_api_address} failed.")

        disconnected(with_error=True)
