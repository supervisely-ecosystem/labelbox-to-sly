import supervisely as sly
from supervisely.app.widgets import Container, Text

info_text = Text()
info_text.hide()
container = Container(widgets=[info_text])
