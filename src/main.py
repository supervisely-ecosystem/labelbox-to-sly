import supervisely as sly
from supervisely.app.widgets import Container, Button

import src.globals as g
import src.ui.copying as copying
import src.ui.keys as keys
import src.ui.selection as selection
import src.ui.info as info

btn = Button("TEST ERROR")

layout = Container(widgets=[btn, keys.card, selection.card, copying.card, info.container])

app = sly.Application(layout=layout)


@btn.click
def set_task_output() -> None:
    """test exception"""
    g.api.task.set_output_error(g.STATE.task_id, "Test exceptions", show_logs=False)
    app.stop()