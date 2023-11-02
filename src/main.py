import supervisely as sly
from supervisely.app.widgets import Container

import src.ui.copying as copying
import src.ui.keys as keys
import src.ui.selection as selection
import src.ui.info as info

layout = Container(widgets=[keys.card, selection.card, copying.card, info.container])

app = sly.Application(layout=layout)
