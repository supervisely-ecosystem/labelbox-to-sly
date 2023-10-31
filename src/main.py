import supervisely as sly
from supervisely.app.widgets import Container

import src.ui.copying as copying
import src.ui.keys as keys
import src.ui.selection as selection

layout = Container(widgets=[keys.card, selection.card, copying.card])

app = sly.Application(layout=layout)
